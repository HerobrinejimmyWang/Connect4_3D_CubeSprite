package com.cubesprite.mobile

import java.util.Random
import kotlin.math.pow
import kotlin.math.sqrt

data class Prediction(val policy: DoubleArray, val value: Double)

fun interface Predictor {
    fun predict(canonicalBoard: IntArray): Prediction
}

data class SearchResult(
    val action: Int,
    val policy: DoubleArray,
    val value: Double,
)

class Mcts(
    private val predictor: Predictor,
    simulations: Int = 128,
    private val cpuct: Double = 1.0,
    private val temperature: Double = 0.4,
    seed: Long? = null,
) {
    private val simulations = simulations.coerceAtLeast(1)
    private val random = if (seed == null) Random() else Random(seed)
    private val priors = HashMap<String, DoubleArray>()
    private val counts = HashMap<String, IntArray>()
    private val values = HashMap<String, DoubleArray>()

    fun run(board: IntArray, player: Int): SearchResult {
        GameRules.validateBoard(board)
        require(player == 1 || player == -1)
        val legal = GameRules.legalMask(board)
        require(legal.any { it }) { "MCTS cannot search a position without legal moves." }

        forcedTacticalAction(board, player)?.let { (action, value) ->
            return SearchResult(
                action,
                DoubleArray(GameRules.ACTION_DIM) { if (it == action) 1.0 else 0.0 },
                value,
            )
        }

        val root = key(board, player)
        expand(board, player)
        repeat(simulations) { simulate(board, player) }

        val rootCounts = counts.getValue(root)
        val policy = countsToPolicy(rootCounts, legal)
        val action = if (temperature > 0.0) sample(policy) else policy.indices.maxBy { policy[it] }
        var weightedValue = 0.0
        var visits = 0
        val rootValues = values.getValue(root)
        for (index in rootCounts.indices) {
            if (rootCounts[index] > 0) {
                weightedValue += rootValues[index]
                visits += rootCounts[index]
            }
        }
        val value = if (visits > 0) {
            weightedValue / visits
        } else {
            predictor.predict(GameRules.canonical(board, player)).value
        }
        return SearchResult(action, policy, value.coerceIn(-1.0, 1.0))
    }

    private fun simulate(board: IntArray, player: Int): Double {
        GameRules.terminalValue(board, player)?.let { return it }
        val key = key(board, player)
        if (!priors.containsKey(key)) return expand(board, player)

        val legal = GameRules.legalMask(board)
        val nodeCounts = counts.getValue(key)
        val nodeValues = values.getValue(key)
        val prior = priors.getValue(key)
        val visitRoot = sqrt(nodeCounts.sum().toDouble() + 1.0)
        var bestAction = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (action in 0 until GameRules.ACTION_DIM) {
            if (!legal[action]) continue
            val q = if (nodeCounts[action] > 0) {
                nodeValues[action] / nodeCounts[action]
            } else {
                0.0
            }
            val score = q + cpuct * prior[action] * visitRoot / (1.0 + nodeCounts[action])
            if (score > bestScore) {
                bestScore = score
                bestAction = action
            }
        }
        check(bestAction >= 0)
        val nextBoard = GameRules.apply(board, player, bestAction)
        val value = -simulate(nextBoard, -player)
        nodeCounts[bestAction] += 1
        nodeValues[bestAction] += value
        return value
    }

    private fun expand(board: IntArray, player: Int): Double {
        val key = key(board, player)
        val prediction = predictor.predict(GameRules.canonical(board, player))
        require(prediction.policy.size == GameRules.ACTION_DIM) {
            "Model policy has ${prediction.policy.size} actions; expected ${GameRules.ACTION_DIM}."
        }
        require(prediction.value.isFinite()) { "Model value output is not finite." }
        priors[key] = TensorMath.maskAndNormalize(prediction.policy, GameRules.legalMask(board))
        counts[key] = IntArray(GameRules.ACTION_DIM)
        values[key] = DoubleArray(GameRules.ACTION_DIM)
        return prediction.value.coerceIn(-1.0, 1.0)
    }

    private fun countsToPolicy(nodeCounts: IntArray, legal: BooleanArray): DoubleArray {
        if (temperature <= 0.0) {
            val result = DoubleArray(GameRules.ACTION_DIM)
            var best = -1
            for (action in nodeCounts.indices) {
                if (legal[action] && (best < 0 || nodeCounts[action] > nodeCounts[best])) best = action
            }
            result[best] = 1.0
            return result
        }
        val exponent = 1.0 / temperature.coerceAtLeast(1e-6)
        val weights = DoubleArray(GameRules.ACTION_DIM)
        for (action in weights.indices) {
            if (legal[action]) weights[action] = nodeCounts[action].toDouble().coerceAtLeast(1e-12).pow(exponent)
        }
        return TensorMath.maskAndNormalize(weights, legal)
    }

    private fun forcedTacticalAction(board: IntArray, player: Int): Pair<Int, Double>? {
        val legal = GameRules.legalActions(board)
        for (action in legal) {
            if (GameRules.hasWin(GameRules.apply(board, player, action), player)) {
                return action to 1.0
            }
        }
        for (action in legal) {
            if (GameRules.hasWin(GameRules.apply(board, -player, action), -player)) {
                val value = predictor.predict(GameRules.canonical(board, player)).value
                return action to value.coerceIn(-1.0, 1.0)
            }
        }
        return null
    }

    private fun sample(policy: DoubleArray): Int {
        val draw = random.nextDouble()
        var cumulative = 0.0
        var fallback = 0
        for (index in policy.indices) {
            if (policy[index] > 0.0) fallback = index
            cumulative += policy[index]
            if (draw <= cumulative) return index
        }
        return fallback
    }

    private fun key(board: IntArray, player: Int): String = buildString(GameRules.ACTION_DIM + 1) {
        append(if (player == 1) 'R' else 'B')
        for (cell in board) append((cell + 1).digitToChar())
    }
}
