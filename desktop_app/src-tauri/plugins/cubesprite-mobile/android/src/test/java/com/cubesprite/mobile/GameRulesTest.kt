package com.cubesprite.mobile

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GameRulesTest {
    @Test
    fun actionMappingAndGravityMatchDesktopRules() {
        assertEquals(Move(0, 0, 0, 0), GameRules.actionToMove(0))
        assertEquals(Move(149, 5, 4, 4), GameRules.actionToMove(149))
        assertEquals(149, GameRules.coordsToAction(5, 4, 4))

        var board = GameRules.emptyBoard()
        val initial = GameRules.legalActions(board)
        assertEquals(25, initial.size)
        assertTrue(initial.all { GameRules.actionToMove(it).layer == 0 })

        board = GameRules.apply(board, 1, GameRules.coordsToAction(0, 2, 3))
        val next = GameRules.legalMask(board)
        assertFalse(next[GameRules.coordsToAction(0, 2, 3)])
        assertTrue(next[GameRules.coordsToAction(1, 2, 3)])
    }

    @Test
    fun winningLineCoversAllThirteenCanonicalAxes() {
        assertEquals(13, GameRules.winDirections.size)
        for (direction in GameRules.winDirections) {
            val startLayer = when {
                direction.layer < 0 -> 3
                direction.layer > 0 -> 0
                else -> 1
            }
            val startRow = when {
                direction.row < 0 -> 3
                direction.row > 0 -> 0
                else -> 1
            }
            val startCol = when {
                direction.col < 0 -> 3
                direction.col > 0 -> 0
                else -> 1
            }
            val board = GameRules.emptyBoard()
            val expected = ArrayList<Int>()
            repeat(GameRules.CONNECT_N) { step ->
                val action = GameRules.coordsToAction(
                    startLayer + step * direction.layer,
                    startRow + step * direction.row,
                    startCol + step * direction.col,
                )
                board[action] = 1
                expected += action
            }
            assertTrue("direction=$direction", GameRules.hasWin(board, 1))
            assertEquals(
                "direction=$direction",
                expected,
                GameRules.findWinningLine(board, 1).map { it.action },
            )
        }
    }

    @Test
    fun canonicalTensorEncodingUsesNcdhwChannelPlanes() {
        val board = GameRules.emptyBoard()
        board[0] = 1
        board[1] = -1
        val encoded = TensorMath.encodeBoard(board, currentPlayer = -1)
        assertEquals(GameRules.ACTION_DIM * 2, encoded.size)
        assertEquals(1f, encoded[1])
        assertEquals(1f, encoded[GameRules.ACTION_DIM])
        assertEquals(2, encoded.count { it == 1f })

        val uniform = TensorMath.softmax(FloatArray(GameRules.ACTION_DIM))
        assertEquals(1.0, uniform.sum(), 1e-12)
        assertArrayEquals(
            DoubleArray(GameRules.ACTION_DIM) { 1.0 / GameRules.ACTION_DIM },
            uniform,
            1e-12,
        )
    }
}
