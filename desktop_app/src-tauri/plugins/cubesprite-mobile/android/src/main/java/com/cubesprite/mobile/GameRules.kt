package com.cubesprite.mobile

data class Move(
    val action: Int,
    val layer: Int,
    val row: Int,
    val col: Int,
    val player: Int? = null,
) {
    fun toMap(includePlayer: Boolean = player != null): Map<String, Any> = buildMap {
        put("action", action)
        put("layer", layer)
        put("row", row)
        put("col", col)
        if (includePlayer && player != null) put("player", player)
    }

    fun toCoordinateMap(): Map<String, Any> = mapOf(
        "layer" to layer,
        "row" to row,
        "col" to col,
    )
}

/**
 * Pure Kotlin rules for the fixed 6 x 5 x 5 gravity board.
 *
 * The flat array uses the same action ordering as the Python backend and ONNX
 * policy head: layer * 25 + row * 5 + column.
 */
object GameRules {
    const val LAYERS = 6
    const val SIZE = 5
    const val CONNECT_N = 4
    const val CELLS_PER_LAYER = SIZE * SIZE
    const val ACTION_DIM = LAYERS * CELLS_PER_LAYER

    data class Direction(val layer: Int, val row: Int, val col: Int)

    val winDirections: List<Direction> = buildList {
        for (dl in 0..1) {
            for (dr in -1..1) {
                for (dc in -1..1) {
                    if (dl == 0 && dr == 0 && dc == 0) continue
                    if (dl == 0 && dr < 0) continue
                    if (dl == 0 && dr == 0 && dc < 0) continue
                    add(Direction(dl, dr, dc))
                }
            }
        }
    }

    init {
        check(winDirections.size == 13)
    }

    fun emptyBoard(): IntArray = IntArray(ACTION_DIM)

    fun actionToMove(action: Int, player: Int? = null): Move {
        require(action in 0 until ACTION_DIM) {
            "Action $action out of range [0, ${ACTION_DIM - 1}]."
        }
        val layer = action / CELLS_PER_LAYER
        val remainder = action % CELLS_PER_LAYER
        return Move(action, layer, remainder / SIZE, remainder % SIZE, player)
    }

    fun coordsToAction(layer: Int, row: Int, col: Int): Int {
        require(isInside(layer, row, col)) {
            "Coordinates ($layer, $row, $col) are outside ($LAYERS, $SIZE, $SIZE)."
        }
        return layer * CELLS_PER_LAYER + row * SIZE + col
    }

    fun legalActions(board: IntArray): IntArray {
        validateBoard(board)
        val actions = IntArray(SIZE * SIZE)
        var count = 0
        for (row in 0 until SIZE) {
            for (col in 0 until SIZE) {
                for (layer in 0 until LAYERS) {
                    val action = coordsToAction(layer, row, col)
                    if (board[action] == 0) {
                        if (layer == 0 || board[action - CELLS_PER_LAYER] != 0) {
                            actions[count++] = action
                        }
                        break
                    }
                }
            }
        }
        return actions.copyOf(count)
    }

    fun legalMask(board: IntArray): BooleanArray {
        val mask = BooleanArray(ACTION_DIM)
        for (action in legalActions(board)) mask[action] = true
        return mask
    }

    fun apply(board: IntArray, player: Int, action: Int): IntArray {
        validateBoard(board)
        require(player == 1 || player == -1) { "Player must be +1 or -1, got $player." }
        val move = actionToMove(action)
        require(board[action] == 0) {
            "Action $action targets occupied position (${move.layer}, ${move.row}, ${move.col})."
        }
        require(move.layer == 0 || board[action - CELLS_PER_LAYER] != 0) {
            "Action $action violates gravity at (${move.layer}, ${move.row}, ${move.col})."
        }
        return board.copyOf().also { it[action] = player }
    }

    fun canonical(board: IntArray, player: Int): IntArray {
        validateBoard(board)
        require(player == 1 || player == -1) { "Player must be +1 or -1, got $player." }
        return IntArray(ACTION_DIM) { board[it] * player }
    }

    fun hasWin(board: IntArray, player: Int): Boolean =
        findWinningLine(board, player).isNotEmpty()

    fun findWinningLine(board: IntArray, player: Int): List<Move> {
        validateBoard(board)
        require(player == 1 || player == -1) { "Player must be +1 or -1, got $player." }
        for (layer in 0 until LAYERS) {
            for (row in 0 until SIZE) {
                for (col in 0 until SIZE) {
                    if (board[coordsToAction(layer, row, col)] != player) continue
                    for (direction in winDirections) {
                        val previousLayer = layer - direction.layer
                        val previousRow = row - direction.row
                        val previousCol = col - direction.col
                        if (
                            isInside(previousLayer, previousRow, previousCol) &&
                            board[coordsToAction(previousLayer, previousRow, previousCol)] == player
                        ) {
                            continue
                        }
                        val cells = ArrayList<Move>(CONNECT_N)
                        var nextLayer = layer
                        var nextRow = row
                        var nextCol = col
                        while (
                            isInside(nextLayer, nextRow, nextCol) &&
                            board[coordsToAction(nextLayer, nextRow, nextCol)] == player
                        ) {
                            cells += actionToMove(
                                coordsToAction(nextLayer, nextRow, nextCol),
                                player,
                            )
                            nextLayer += direction.layer
                            nextRow += direction.row
                            nextCol += direction.col
                        }
                        if (cells.size >= CONNECT_N) return cells.take(CONNECT_N)
                    }
                }
            }
        }
        return emptyList()
    }

    /**
     * Value from [player]'s perspective, or null for a non-terminal position.
     */
    fun terminalValue(board: IntArray, player: Int): Double? {
        if (hasWin(board, -player)) return -1.0
        if (hasWin(board, player)) return 1.0
        return if (legalActions(board).isEmpty()) 0.0 else null
    }

    fun validateBoard(board: IntArray) {
        require(board.size == ACTION_DIM) {
            "Board must contain $ACTION_DIM cells, got ${board.size}."
        }
        require(board.all { it in -1..1 }) {
            "Board cells must contain only -1, 0, or +1."
        }
    }

    private fun isInside(layer: Int, row: Int, col: Int): Boolean =
        layer in 0 until LAYERS && row in 0 until SIZE && col in 0 until SIZE
}
