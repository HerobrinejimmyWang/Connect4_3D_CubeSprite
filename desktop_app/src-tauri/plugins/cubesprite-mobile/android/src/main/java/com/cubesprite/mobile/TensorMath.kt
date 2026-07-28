package com.cubesprite.mobile

import kotlin.math.exp

object TensorMath {
    /**
     * Encode NCDHW [1, 2, 6, 5, 5] after canonicalising to the player to move.
     */
    fun encodeBoard(board: IntArray, currentPlayer: Int): FloatArray {
        GameRules.validateBoard(board)
        require(currentPlayer == 1 || currentPlayer == -1)
        val encoded = FloatArray(GameRules.ACTION_DIM * 2)
        for (index in board.indices) {
            val canonical = board[index] * currentPlayer
            if (canonical > 0) encoded[index] = 1f
            if (canonical < 0) encoded[GameRules.ACTION_DIM + index] = 1f
        }
        return encoded
    }

    fun softmax(logits: FloatArray): DoubleArray {
        require(logits.isNotEmpty())
        require(logits.all { it.isFinite() }) { "Model policy output contains a non-finite value." }
        val maximum = logits.maxOrNull()!!.toDouble()
        val probabilities = DoubleArray(logits.size) {
            exp((logits[it].toDouble() - maximum).coerceIn(-80.0, 0.0))
        }
        val total = probabilities.sum()
        if (!total.isFinite() || total <= 0.0) {
            return DoubleArray(logits.size) { 1.0 / logits.size }
        }
        for (index in probabilities.indices) probabilities[index] /= total
        return probabilities
    }

    fun maskAndNormalize(policy: DoubleArray, legal: BooleanArray): DoubleArray {
        require(policy.size == GameRules.ACTION_DIM)
        require(legal.size == GameRules.ACTION_DIM)
        val masked = DoubleArray(GameRules.ACTION_DIM)
        var total = 0.0
        var legalCount = 0
        for (index in masked.indices) {
            if (legal[index]) {
                legalCount += 1
                val weight = policy[index].takeIf { it.isFinite() && it > 0.0 } ?: 0.0
                masked[index] = weight
                total += weight
            }
        }
        require(legalCount > 0) { "Cannot normalize a policy without legal actions." }
        if (total <= 0.0 || !total.isFinite()) {
            val uniform = 1.0 / legalCount
            for (index in masked.indices) if (legal[index]) masked[index] = uniform
        } else {
            for (index in masked.indices) masked[index] /= total
        }
        return masked
    }
}
