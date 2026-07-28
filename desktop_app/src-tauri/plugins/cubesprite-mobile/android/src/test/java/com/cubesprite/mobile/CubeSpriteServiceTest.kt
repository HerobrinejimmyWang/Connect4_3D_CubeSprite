package com.cubesprite.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class CubeSpriteServiceTest {
    private class FakeRuntime : ModelRuntime {
        override val specs = listOf(
            spec("cubesprite_v3_mini", "CubeSprite V3 mini", 128),
            spec("cubesprite_v3", "CubeSprite V3", 128),
        )

        override fun listModels(): List<Map<String, Any?>> =
            specs.map { it.toMap(available = true, unavailableReason = null) }

        override fun getSpec(modelId: String): ModelSpec =
            specs.singleOrNull { it.id == modelId }
                ?: throw IllegalArgumentException("Unknown model id: $modelId")

        override fun predict(modelId: String, canonicalBoard: IntArray): Prediction =
            Prediction(DoubleArray(GameRules.ACTION_DIM) { 1.0 / GameRules.ACTION_DIM }, 0.0)

        override fun close() = Unit

        companion object {
            private fun spec(id: String, name: String, simulations: Int) = ModelSpec(
                id = id,
                displayName = name,
                modelPath = "models/$id.onnx",
                architecture = "gravity_resnet_v1",
                boardLayers = GameRules.LAYERS,
                boardSize = GameRules.SIZE,
                inputChannels = 2,
                actionDim = GameRules.ACTION_DIM,
                artifactSha256 = "0".repeat(64),
                sourceIteration = 1,
                defaultMctsSims = simulations,
                defaultTemperature = 0.4,
                description = mapOf("zh" to name, "en" to name),
            )
        }
    }

    @Test
    fun initializeDeclaresReplayCapabilityAndMiniDefaults() {
        CubeSpriteService(FakeRuntime()).use { service ->
            val initialized = service.initialize()
            assertEquals(mapOf("replay" to false), initialized["capabilities"])
            assertEquals(listOf(32, 64, 128, 256, 512), initialized["mcts_options"])
            val models = initialized["models"] as List<*>
            assertEquals("cubesprite_v3_mini", (models.first() as Map<*, *>)["id"])
            val settings = initialized["settings"] as Map<*, *>
            val roles = settings["roles"] as Map<*, *>
            for (role in listOf("combat", "hint", "win_rate")) {
                val config = roles[role] as Map<*, *>
                assertEquals("cubesprite_v3_mini", config["model_id"])
                assertEquals(128, config["mcts_sims"])
                assertEquals(0.4, config["temperature"])
            }
        }
    }

    @Test
    fun accepts512MctsAndRejectsValuesOutsideTheMobileOptions() {
        CubeSpriteService(FakeRuntime()).use { service ->
            val state = service.handle(
                "game.new",
                mapOf("mode" to "pvp", "human_player" to 1),
            ) as Map<*, *>
            val token = mapOf(
                "session_id" to state["session_id"],
                "expected_revision" to state["revision"],
            )
            val result = service.handle(
                "analysis.hint",
                token + mapOf(
                    "ai" to mapOf(
                        "model_id" to "cubesprite_v3_mini",
                        "mcts_sims" to 512,
                        "temperature" to 0.4,
                    ),
                ),
            ) as Map<*, *>
            assertEquals(state["revision"], result["for_revision"])

            try {
                service.handle(
                    "analysis.hint",
                    token + mapOf(
                        "ai" to mapOf(
                            "model_id" to "cubesprite_v3_mini",
                            "mcts_sims" to 1024,
                            "temperature" to 0.4,
                        ),
                    ),
                )
                fail("Expected invalid MCTS settings")
            } catch (error: ServiceError) {
                assertEquals("INVALID_MCTS", error.code)
            }
        }
    }

    @Test
    fun moveUndoRestartAndStaleRevisionAreAuthoritative() {
        CubeSpriteService(FakeRuntime()).use { service ->
            val initial = service.handle(
                "game.new",
                mapOf("mode" to "pvp", "human_player" to 1),
            ) as Map<*, *>
            val token = mapOf(
                "session_id" to initial["session_id"],
                "expected_revision" to initial["revision"],
            )
            val moved = service.handle(
                "game.move",
                token + mapOf("layer" to 0, "row" to 0, "col" to 0),
            ) as Map<*, *>
            assertEquals(1, moved["move_count"])
            assertTrue(moved["can_undo"] as Boolean)
            assertNotEquals(initial["revision"], moved["revision"])

            try {
                service.handle(
                    "game.move",
                    token + mapOf("layer" to 0, "row" to 0, "col" to 1),
                )
                fail("Expected stale revision")
            } catch (error: ServiceError) {
                assertEquals("STALE_REVISION", error.code)
            }

            val undone = service.handle(
                "game.undo",
                mapOf(
                    "session_id" to moved["session_id"],
                    "expected_revision" to moved["revision"],
                ),
            ) as Map<*, *>
            assertEquals(0, undone["move_count"])
            assertFalse(undone["can_undo"] as Boolean)

            val restarted = service.handle(
                "game.restart",
                mapOf(
                    "session_id" to undone["session_id"],
                    "expected_revision" to undone["revision"],
                ),
            ) as Map<*, *>
            assertEquals(0, restarted["move_count"])
            assertEquals(undone["session_id"], restarted["session_id"])
            assertNotEquals(undone["revision"], restarted["revision"])
        }
    }
}
