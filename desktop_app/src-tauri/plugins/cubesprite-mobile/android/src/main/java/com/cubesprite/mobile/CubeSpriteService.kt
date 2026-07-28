package com.cubesprite.mobile

import android.content.Context
import java.util.UUID

class ServiceError(
    val code: String,
    override val message: String,
    val details: Any? = null,
) : RuntimeException(message)

data class AiConfig(
    val modelId: String,
    val mctsSims: Int,
    val temperature: Double,
) {
    fun toMap(): Map<String, Any> = mapOf(
        "model_id" to modelId,
        "mcts_sims" to mctsSims,
        "temperature" to temperature,
    )
}

class CubeSpriteService internal constructor(
    private val runtime: ModelRuntime,
) : AutoCloseable {
    constructor(context: Context) : this(OrtModelRuntime(context))

    companion object {
        private val MCTS_OPTIONS = listOf(32, 64, 128, 256, 512)
        private const val DEFAULT_MODEL_ID = "cubesprite_v3_mini"
        private const val DEFAULT_MCTS_SIMS = 128
        private const val DEFAULT_TEMPERATURE = 0.4
    }

    private val defaultAi = AiConfig(DEFAULT_MODEL_ID, DEFAULT_MCTS_SIMS, DEFAULT_TEMPERATURE)
    private var revision = 0
    private var sessionId = ""
    private var mode = "pvp"
    private var humanPlayer = 1
    private var board = GameRules.emptyBoard()
    private var currentPlayer = 1
    private val history = ArrayList<Move>()
    private var status = "playing"
    private var winner: Int? = null
    private var winningLine = emptyList<Move>()
    private var lastMove: Move? = null
    private var historyFloor = 0
    private var sessionOriginHistory = emptyList<Move>()

    init {
        newSession("pvp", 1)
    }

    fun handle(command: String, params: Map<String, Any?>): Any = when (command) {
        "system.initialize" -> initialize()
        "system.ping" -> mapOf("pong" to true, "version" to "0.1.0-android")
        "models.list" -> mapOf("models" to runtime.listModels())
        "settings.get" -> settingsSnapshot()
        "game.new" -> newGame(params)
        "game.state" -> snapshot()
        "game.move" -> move(params)
        "game.ai_move" -> aiMove(params)
        "game.undo" -> undo(params)
        "game.restart" -> restart(params)
        "analysis.hint" -> hint(params)
        "analysis.win_rate" -> winRate(params)
        "replay.list" -> mapOf("replays" to emptyList<Any>())
        "replay.save",
        "replay.open",
        "replay.delete",
        "replay.export",
        "replay.import",
        "replay.analyze",
        "replay.continue",
        -> throw ServiceError(
            "UNSUPPORTED_ON_ANDROID",
            "Replay storage and replay analysis are not included in the lightweight Android build.",
        )
        else -> throw ServiceError("UNKNOWN_COMMAND", "Unknown command: $command")
    }

    fun initialize(): Map<String, Any> = linkedMapOf(
        "backend_version" to "0.1.0-android",
        "protocol_version" to 1,
        "board" to mapOf(
            "layers" to GameRules.LAYERS,
            "size" to GameRules.SIZE,
            "connect_n" to GameRules.CONNECT_N,
        ),
        "mcts_options" to MCTS_OPTIONS,
        "models" to runtime.listModels(),
        "settings" to settingsSnapshot(),
        "capabilities" to mapOf("replay" to false),
        "state" to snapshot(),
    )

    fun snapshot(): Map<String, Any?> {
        val undoable = history.drop(historyFloor)
        val legalMoves = if (status == "playing") {
            GameRules.legalActions(board).map { GameRules.actionToMove(it).toMap(false) }
        } else {
            emptyList()
        }
        val canUndo = if (mode == "pvp") {
            undoable.isNotEmpty()
        } else {
            undoable.any { it.player == humanPlayer }
        }
        return linkedMapOf(
            "session_id" to sessionId,
            "revision" to revision,
            "mode" to mode,
            "human_player" to humanPlayer,
            "board" to nestedBoard(board),
            "current_player" to currentPlayer,
            "move_count" to history.size,
            "status" to status,
            "winner" to winner,
            "last_move" to lastMove?.toMap(),
            "winning_line" to winningLine.map { it.toCoordinateMap() },
            "legal_moves" to legalMoves,
            "can_undo" to canUndo,
        )
    }

    override fun close() {
        runtime.close()
    }

    private fun settingsSnapshot(): Map<String, Any> = mapOf(
        "roles" to mapOf(
            "combat" to defaultAi.toMap(),
            "hint" to defaultAi.toMap(),
            "win_rate" to defaultAi.toMap(),
        ),
        "preload_hint" to false,
    )

    private fun newGame(params: Map<String, Any?>): Map<String, Any?> {
        val nextMode = params["mode"]?.toString() ?: "pvp"
        val nextHuman = intParam(params["human_player"] ?: 1, "human_player")
        if (nextMode !in setOf("pvp", "pvai")) {
            throw ServiceError("INVALID_MODE", "Unsupported game mode: $nextMode")
        }
        if (nextHuman != 1 && nextHuman != -1) {
            throw ServiceError("INVALID_PLAYER", "human_player must be +1 or -1")
        }
        newSession(nextMode, nextHuman)
        return snapshot()
    }

    private fun move(params: Map<String, Any?>): Map<String, Any?> {
        checkConcurrency(params, required = true)
        if (status != "playing") {
            throw ServiceError("GAME_FINISHED", "The game has already ended.")
        }
        if (mode == "pvai" && currentPlayer != humanPlayer) {
            throw ServiceError("NOT_HUMAN_TURN", "It is the AI player's turn.")
        }
        val layer = requiredInt(params, "layer")
        val row = requiredInt(params, "row")
        val col = requiredInt(params, "col")
        val action = try {
            GameRules.coordsToAction(layer, row, col)
        } catch (error: IllegalArgumentException) {
            throw ServiceError("ILLEGAL_MOVE", error.message ?: "Illegal move.", error.message)
        }
        applyAction(action, currentPlayer, bumpRevision = true)
        return snapshot()
    }

    private fun aiMove(params: Map<String, Any?>): Map<String, Any?> {
        checkAnalysisReady(params, requireAiTurn = true)
        val capturedBoard = board.copyOf()
        val player = currentPlayer
        val capturedRevision = revision
        val capturedSession = sessionId
        val ai = aiConfig(params["ai"])
        val result = search(capturedBoard, player, ai)
        assertFresh(capturedSession, capturedRevision)
        applyAction(result.action, player, bumpRevision = true)
        return snapshot().toMutableMap().apply {
            put(
                "analysis",
                mapOf("value" to result.value, "policy" to result.policy.toList()),
            )
        }
    }

    private fun hint(params: Map<String, Any?>): Map<String, Any> {
        checkAnalysisReady(params)
        val capturedBoard = board.copyOf()
        val player = currentPlayer
        val capturedRevision = revision
        val capturedSession = sessionId
        val result = search(capturedBoard, player, aiConfig(params["ai"]))
        assertFresh(capturedSession, capturedRevision)
        return mapOf(
            "session_id" to capturedSession,
            "for_revision" to capturedRevision,
            "move" to GameRules.actionToMove(result.action).toMap(false),
            "value" to result.value,
        )
    }

    private fun winRate(params: Map<String, Any?>): Map<String, Any> {
        checkAnalysisReady(params)
        val capturedBoard = board.copyOf()
        val player = currentPlayer
        val capturedRevision = revision
        val capturedSession = sessionId
        val result = search(capturedBoard, player, aiConfig(params["ai"]))
        assertFresh(capturedSession, capturedRevision)
        val currentProbability = ((result.value + 1.0) / 2.0).coerceIn(0.0, 1.0)
        val red = if (player == 1) currentProbability else 1.0 - currentProbability
        return mapOf(
            "session_id" to capturedSession,
            "for_revision" to capturedRevision,
            "red" to red,
            "blue" to 1.0 - red,
            "estimate" to "model_mcts",
        )
    }

    private fun undo(params: Map<String, Any?>): Map<String, Any?> {
        checkConcurrency(params, required = true)
        if (history.size <= historyFloor) return snapshot()
        if (mode == "pvp") {
            history.removeAt(history.lastIndex)
        } else {
            val undoable = history.drop(historyFloor)
            if (undoable.none { it.player == humanPlayer }) return snapshot()
            if (history.size > historyFloor && history.last().player != humanPlayer) {
                history.removeAt(history.lastIndex)
            }
            if (history.size > historyFloor && history.last().player == humanPlayer) {
                history.removeAt(history.lastIndex)
            }
        }
        rebuildFromHistory()
        revision += 1
        return snapshot()
    }

    private fun restart(params: Map<String, Any?>): Map<String, Any?> {
        checkConcurrency(params, required = true)
        restoreHistory(sessionOriginHistory)
        historyFloor = history.size
        revision += 1
        return snapshot()
    }

    private fun checkAnalysisReady(params: Map<String, Any?>, requireAiTurn: Boolean = false) {
        checkConcurrency(params, required = true)
        if (status != "playing") {
            throw ServiceError("GAME_FINISHED", "The game has already ended.")
        }
        if (requireAiTurn && (mode != "pvai" || currentPlayer == humanPlayer)) {
            throw ServiceError("NOT_AI_TURN", "It is not the AI player's turn.")
        }
    }

    private fun aiConfig(raw: Any?): AiConfig {
        if (raw == null) return defaultAi
        val map = raw as? Map<*, *> ?: throw ServiceError(
            "INVALID_AI_SETTINGS",
            "ai must be a JSON object.",
        )
        val unknown = map.keys.map { it.toString() }.filter {
            it !in setOf("model_id", "mcts_sims", "temperature")
        }
        if (unknown.isNotEmpty()) {
            throw ServiceError(
                "INVALID_AI_SETTINGS",
                "Unknown AI setting(s): ${unknown.sorted().joinToString()}",
            )
        }
        val modelId = map["model_id"]?.toString() ?: defaultAi.modelId
        try {
            runtime.getSpec(modelId)
        } catch (error: IllegalArgumentException) {
            throw ServiceError("MODEL_UNAVAILABLE", error.message ?: "Unknown model.")
        }
        val simulations = intParam(map["mcts_sims"] ?: defaultAi.mctsSims, "mcts_sims")
        if (simulations !in MCTS_OPTIONS) {
            throw ServiceError(
                "INVALID_MCTS",
                "MCTS simulations must be one of $MCTS_OPTIONS",
            )
        }
        val temperature = numberParam(
            map["temperature"] ?: defaultAi.temperature,
            "temperature",
        )
        if (!temperature.isFinite() || temperature !in 0.0..5.0) {
            throw ServiceError(
                "INVALID_TEMPERATURE",
                "temperature must be a number from 0 to 5.",
            )
        }
        return AiConfig(modelId, simulations, kotlin.math.round(temperature * 10.0) / 10.0)
    }

    private fun search(board: IntArray, player: Int, ai: AiConfig): SearchResult = try {
        Mcts(
            predictor = Predictor { canonical -> runtime.predict(ai.modelId, canonical) },
            simulations = ai.mctsSims,
            temperature = ai.temperature,
        ).run(board, player)
    } catch (error: ServiceError) {
        throw error
    } catch (error: Exception) {
        throw ServiceError(
            "INFERENCE_FAILED",
            error.message ?: "ONNX inference failed.",
            error.javaClass.simpleName,
        )
    }

    private fun newSession(nextMode: String, nextHumanPlayer: Int) {
        sessionId = UUID.randomUUID().toString()
        mode = nextMode
        humanPlayer = nextHumanPlayer
        sessionOriginHistory = emptyList()
        historyFloor = 0
        resetPosition()
        revision += 1
    }

    private fun resetPosition() {
        board = GameRules.emptyBoard()
        currentPlayer = 1
        history.clear()
        status = "playing"
        winner = null
        winningLine = emptyList()
        lastMove = null
    }

    private fun applyAction(action: Int, player: Int, bumpRevision: Boolean) {
        if (player != currentPlayer) {
            throw ServiceError(
                "WRONG_PLAYER",
                "The move player does not match the authoritative turn.",
            )
        }
        board = try {
            GameRules.apply(board, player, action)
        } catch (error: IllegalArgumentException) {
            throw ServiceError("ILLEGAL_MOVE", error.message ?: "Illegal move.")
        }
        val move = GameRules.actionToMove(action, player)
        history += move
        lastMove = move
        val line = GameRules.findWinningLine(board, player)
        when {
            line.isNotEmpty() -> {
                status = "won"
                winner = player
                winningLine = line
            }
            GameRules.legalActions(board).isEmpty() -> {
                status = "draw"
                winner = 0
            }
            else -> currentPlayer = -player
        }
        if (bumpRevision) revision += 1
    }

    private fun rebuildFromHistory() {
        restoreHistory(history.toList())
    }

    private fun restoreHistory(moves: List<Move>) {
        resetPosition()
        for (move in moves) {
            if (status != "playing" || move.player != currentPlayer) {
                throw ServiceError("CORRUPT_HISTORY", "Game history is inconsistent.")
            }
            applyAction(move.action, move.player, bumpRevision = false)
        }
    }

    private fun checkConcurrency(params: Map<String, Any?>, required: Boolean) {
        val requestedSession = params["session_id"]
        val requestedRevision = params["expected_revision"]
        if (required && (requestedSession == null || requestedRevision == null)) {
            throw ServiceError(
                "MISSING_CONCURRENCY_TOKEN",
                "session_id and expected_revision are required for this command.",
            )
        }
        if (requestedSession != null && requestedSession.toString() != sessionId) {
            throw ServiceError("STALE_SESSION", "The game session has changed.")
        }
        if (requestedRevision != null && intParam(requestedRevision, "expected_revision") != revision) {
            throw ServiceError(
                "STALE_REVISION",
                "The game state changed while this request was pending.",
            )
        }
    }

    private fun assertFresh(expectedSession: String, expectedRevision: Int) {
        if (expectedSession != sessionId) {
            throw ServiceError(
                "STALE_SESSION",
                "The game session changed while this request was pending.",
            )
        }
        if (expectedRevision != revision) {
            throw ServiceError(
                "STALE_REVISION",
                "The game state changed while this request was pending.",
            )
        }
    }

    private fun requiredInt(params: Map<String, Any?>, name: String): Int {
        if (!params.containsKey(name)) {
            throw ServiceError("INVALID_PARAMS", "Missing move coordinate: $name")
        }
        return intParam(params[name], name)
    }

    private fun intParam(value: Any?, name: String): Int {
        if (value !is Number) {
            throw ServiceError("INVALID_PARAMS", "$name must be an integer.")
        }
        val number = value.toDouble()
        if (!number.isFinite() || number % 1.0 != 0.0 || number !in Int.MIN_VALUE.toDouble()..Int.MAX_VALUE.toDouble()) {
            throw ServiceError("INVALID_PARAMS", "$name must be an integer.")
        }
        return number.toInt()
    }

    private fun numberParam(value: Any?, name: String): Double {
        if (value !is Number) {
            throw ServiceError("INVALID_PARAMS", "$name must be a number.")
        }
        return value.toDouble()
    }

    private fun nestedBoard(flat: IntArray): List<List<List<Int>>> =
        List(GameRules.LAYERS) { layer ->
            List(GameRules.SIZE) { row ->
                List(GameRules.SIZE) { col ->
                    flat[GameRules.coordsToAction(layer, row, col)]
                }
            }
        }
}
