package com.cubesprite.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.UUID

data class ModelSpec(
    val id: String,
    val displayName: String,
    val modelPath: String,
    val architecture: String,
    val boardLayers: Int,
    val boardSize: Int,
    val inputChannels: Int,
    val actionDim: Int,
    val artifactSha256: String,
    val sourceIteration: Int?,
    val defaultMctsSims: Int,
    val defaultTemperature: Double,
    val description: Map<String, String>,
) {
    fun toMap(available: Boolean, unavailableReason: String?): Map<String, Any?> = linkedMapOf(
        "id" to id,
        "display_name" to displayName,
        "model_path" to modelPath,
        "architecture" to architecture,
        "board_layers" to boardLayers,
        "board_size" to boardSize,
        "input_channels" to inputChannels,
        "action_dim" to actionDim,
        "artifact_sha256" to artifactSha256,
        "source_iteration" to sourceIteration,
        "defaults" to mapOf(
            "mcts_sims" to defaultMctsSims,
            "temperature" to defaultTemperature,
        ),
        "default_mcts_sims" to defaultMctsSims,
        "default_temperature" to defaultTemperature,
        "description" to description,
        "placeholder" to false,
        "available" to available,
        "unavailable_reason" to unavailableReason,
    )
}

interface ModelRuntime : AutoCloseable {
    val specs: List<ModelSpec>
    fun listModels(): List<Map<String, Any?>>
    fun getSpec(modelId: String): ModelSpec
    fun predict(modelId: String, canonicalBoard: IntArray): Prediction
}

class OrtModelRuntime(private val context: Context) : ModelRuntime {
    companion object {
        private val INPUT_SHAPE = longArrayOf(
            1,
            2,
            GameRules.LAYERS.toLong(),
            GameRules.SIZE.toLong(),
            GameRules.SIZE.toLong(),
        )
        private val EXPECTED_IDS = listOf("cubesprite_v3_mini", "cubesprite_v3")
    }

    private data class LoadedModel(
        val spec: ModelSpec,
        val options: OrtSession.SessionOptions,
        val session: OrtSession,
        val inputName: String,
        val policyOutputName: String,
        val valueOutputName: String,
    ) : AutoCloseable {
        override fun close() {
            session.close()
            options.close()
        }
    }

    private val environment = OrtEnvironment.getEnvironment()
    override val specs: List<ModelSpec> = loadRegistry()
    private val specsById = specs.associateBy { it.id }
    private val loadErrors = HashMap<String, String>()
    private var loaded: LoadedModel? = null

    override fun listModels(): List<Map<String, Any?>> = specs.map { spec ->
        val assetPresent = runCatching {
            context.assets.open(spec.modelPath).use { }
            true
        }.getOrDefault(false)
        val reason = loadErrors[spec.id] ?: if (assetPresent) null else "model_file_missing"
        spec.toMap(reason == null, reason)
    }

    override fun getSpec(modelId: String): ModelSpec =
        specsById[modelId] ?: throw IllegalArgumentException("Unknown model id: $modelId")

    @Synchronized
    override fun predict(modelId: String, canonicalBoard: IntArray): Prediction {
        val model = loadModel(modelId)
        val encoded = TensorMath.encodeBoard(canonicalBoard, 1)
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(encoded),
            INPUT_SHAPE,
        ).use { input ->
            model.session.run(mapOf(model.inputName to input)).use { outputs ->
                val policyTensor = outputs.get(model.policyOutputName).orElseThrow {
                    IllegalStateException("ONNX result omitted ${model.policyOutputName}.")
                } as? OnnxTensor ?: error("Policy output is not a tensor.")
                val valueTensor = outputs.get(model.valueOutputName).orElseThrow {
                    IllegalStateException("ONNX result omitted ${model.valueOutputName}.")
                } as? OnnxTensor ?: error("Value output is not a tensor.")
                val logits = flattenFloats(policyTensor.value)
                require(logits.size == GameRules.ACTION_DIM) {
                    "Model policy has ${logits.size} actions; expected ${GameRules.ACTION_DIM}."
                }
                val rawValue = flattenFloats(valueTensor.value).singleOrNull()
                    ?: error("Model value output must contain exactly one number.")
                require(rawValue.isFinite()) { "Model value output is not finite." }
                return Prediction(
                    TensorMath.softmax(logits),
                    rawValue.toDouble().coerceIn(-1.0, 1.0),
                )
            }
        }
    }

    @Synchronized
    override fun close() {
        loaded?.close()
        loaded = null
    }

    @Synchronized
    private fun loadModel(modelId: String): LoadedModel {
        val spec = getSpec(modelId)
        loaded?.takeIf { it.spec.id == spec.id }?.let { return it }
        val previous = loaded
        loaded = null
        previous?.close()
        val next = try {
            createLoadedModel(spec)
        } catch (error: Exception) {
            val reason = "model_load_failed: ${error.message ?: error.javaClass.simpleName}"
            loadErrors[spec.id] = reason
            throw IllegalStateException("Cannot load ${spec.displayName}: ${error.message}", error)
        }
        loaded = next
        loadErrors.remove(spec.id)
        return next
    }

    private fun createLoadedModel(spec: ModelSpec): LoadedModel {
        val modelFile = ensurePrivateModelFile(spec)
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(
                (Runtime.getRuntime().availableProcessors() / 2).coerceIn(1, 4),
            )
            setInterOpNumThreads(1)
        }
        try {
            val session = environment.createSession(modelFile.absolutePath, options)
            try {
                require(session.inputInfo.size == 1) {
                    "Expected exactly one ONNX input."
                }
                require(session.outputInfo.size == 2) {
                    "Expected exactly two ONNX outputs (policy, value)."
                }
                val input = session.inputInfo.entries.single()
                validateInputShape((input.value.info as? TensorInfo)?.shape)
                val policy = session.outputInfo.entries.singleOrNull {
                    lastStaticDimension((it.value.info as? TensorInfo)?.shape) == spec.actionDim.toLong()
                } ?: error("ONNX outputs must expose policy[${spec.actionDim}].")
                val value = session.outputInfo.entries.singleOrNull {
                    lastStaticDimension((it.value.info as? TensorInfo)?.shape) == 1L
                } ?: error("ONNX outputs must expose value[1].")
                require(policy.key != value.key) {
                    "ONNX policy and value outputs must be distinct."
                }
                return LoadedModel(spec, options, session, input.key, policy.key, value.key)
            } catch (error: Exception) {
                session.close()
                throw error
            }
        } catch (error: Exception) {
            options.close()
            throw error
        }
    }

    private fun ensurePrivateModelFile(spec: ModelSpec): File {
        val directory = File(context.noBackupFilesDir, "cubesprite-models")
        require(directory.isDirectory || directory.mkdirs()) {
            "Cannot prepare private model directory ${directory.absolutePath}."
        }
        val destination = File(
            directory,
            "${spec.id}-${spec.artifactSha256.take(12)}.onnx",
        )
        if (destination.isFile && sha256(destination) == spec.artifactSha256) return destination
        if (destination.exists() && !destination.delete()) {
            error("Cannot replace invalid model file ${destination.absolutePath}.")
        }

        val temporary = File(directory, ".${destination.name}.${UUID.randomUUID()}.tmp")
        val digest = MessageDigest.getInstance("SHA-256")
        try {
            context.assets.open(spec.modelPath).use { input ->
                FileOutputStream(temporary).use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            }
            val actual = digest.digest().toHex()
            require(actual == spec.artifactSha256) {
                "ONNX artifact SHA-256 mismatch for ${spec.id}: expected ${spec.artifactSha256}, found $actual."
            }
            require(temporary.renameTo(destination)) {
                "Cannot publish private model file ${destination.absolutePath}."
            }
            return destination
        } finally {
            if (temporary.exists()) temporary.delete()
        }
    }

    private fun loadRegistry(): List<ModelSpec> {
        val payload = context.assets.open("model_registry.json").bufferedReader(Charsets.UTF_8).use {
            JSONObject(it.readText())
        }
        require(payload.getInt("manifest_version") == 1) {
            "Unsupported Android model registry version."
        }
        val entries = payload.getJSONArray("models")
        val models = ArrayList<ModelSpec>(entries.length())
        for (index in 0 until entries.length()) {
            val entry = entries.getJSONObject(index)
            val defaults = entry.getJSONObject("defaults")
            val description = entry.getJSONObject("description")
            models += ModelSpec(
                id = entry.getString("id"),
                displayName = entry.getString("display_name"),
                modelPath = entry.getString("model_path"),
                architecture = entry.getString("architecture"),
                boardLayers = entry.getInt("board_layers"),
                boardSize = entry.getInt("board_size"),
                inputChannels = entry.getInt("input_channels"),
                actionDim = entry.getInt("action_dim"),
                artifactSha256 = entry.getString("artifact_sha256"),
                sourceIteration = if (entry.isNull("source_iteration")) null else entry.getInt("source_iteration"),
                defaultMctsSims = defaults.getInt("mcts_sims"),
                defaultTemperature = defaults.getDouble("temperature"),
                description = mapOf(
                    "zh" to description.getString("zh"),
                    "en" to description.getString("en"),
                ),
            )
        }
        require(models.map { it.id } == EXPECTED_IDS) {
            "Android model registry must contain mini then full CubeSprite V3 only."
        }
        models.forEach { spec ->
            require(spec.architecture == "gravity_resnet_v1")
            require(spec.boardLayers == GameRules.LAYERS)
            require(spec.boardSize == GameRules.SIZE)
            require(spec.inputChannels == 2)
            require(spec.actionDim == GameRules.ACTION_DIM)
            require(spec.artifactSha256.matches(Regex("[0-9a-f]{64}")))
            require(spec.defaultMctsSims in setOf(32, 64, 128, 256, 512))
            require(spec.defaultTemperature in 0.0..5.0)
            require(!spec.modelPath.startsWith("/") && ".." !in spec.modelPath.split('/'))
        }
        return models
    }

    private fun validateInputShape(shape: LongArray?) {
        require(shape != null && shape.size == INPUT_SHAPE.size) {
            "ONNX input rank must be 5."
        }
        for (index in 1 until INPUT_SHAPE.size) {
            require(shape[index] <= 0 || shape[index] == INPUT_SHAPE[index]) {
                "ONNX input shape ${shape.contentToString()} does not match ${INPUT_SHAPE.contentToString()}."
            }
        }
    }

    private fun lastStaticDimension(shape: LongArray?): Long? =
        shape?.lastOrNull()?.takeIf { it > 0 }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().toHex()
    }

    private fun flattenFloats(value: Any?): FloatArray = when (value) {
        is FloatArray -> value
        is Array<*> -> value.flatMap { flattenFloats(it).asIterable() }.toFloatArray()
        is Number -> floatArrayOf(value.toFloat())
        else -> error("Unsupported ONNX tensor value ${value?.javaClass?.name ?: "null"}.")
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
}
