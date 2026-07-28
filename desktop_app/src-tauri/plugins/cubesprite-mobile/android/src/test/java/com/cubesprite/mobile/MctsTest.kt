package com.cubesprite.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MctsTest {
    private val uniformPredictor = Predictor {
        Prediction(
            DoubleArray(GameRules.ACTION_DIM) { 1.0 / GameRules.ACTION_DIM },
            0.25,
        )
    }

    @Test
    fun immediatelyTakesAWinningMove() {
        val board = GameRules.emptyBoard()
        for (col in 0..2) {
            board[GameRules.coordsToAction(0, 0, col)] = 1
        }
        val result = Mcts(uniformPredictor, simulations = 1, seed = 7).run(board, 1)
        assertEquals(GameRules.coordsToAction(0, 0, 3), result.action)
        assertEquals(1.0, result.value, 0.0)
        assertEquals(1.0, result.policy.sum(), 1e-12)
    }

    @Test
    fun immediatelyBlocksAnOpponentWin() {
        val board = GameRules.emptyBoard()
        for (col in 0..2) {
            board[GameRules.coordsToAction(0, 1, col)] = -1
        }
        val result = Mcts(uniformPredictor, simulations = 1, seed = 7).run(board, 1)
        assertEquals(GameRules.coordsToAction(0, 1, 3), result.action)
        assertEquals(0.25, result.value, 0.0)
        assertTrue(GameRules.legalMask(board)[result.action])
    }

    @Test
    fun searchAlwaysReturnsANormalizedLegalPolicy() {
        val board = GameRules.apply(GameRules.emptyBoard(), 1, 0)
        val result = Mcts(
            uniformPredictor,
            simulations = 8,
            temperature = 1.0,
            seed = 11,
        ).run(board, -1)
        val legal = GameRules.legalMask(board)
        assertTrue(legal[result.action])
        assertEquals(1.0, result.policy.sum(), 1e-12)
        assertTrue(result.policy.indices.all { legal[it] || result.policy[it] == 0.0 })
    }
}
