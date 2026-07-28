package com.cubesprite.mobile

import android.app.Activity
import android.content.pm.ActivityInfo
import android.view.View
import android.webkit.WebView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.Insets
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.Plugin
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@InvokeArg
class RequestArgs {
    lateinit var command: String
    var params: Map<String, Any?> = emptyMap()
}

/**
 * One serialized native worker owns the authoritative game and ORT session.
 * This keeps model loading and MCTS off the WebView/UI thread and prevents
 * concurrent mutations from racing with one another.
 */
@TauriPlugin
class CubeSpritePlugin(private val activity: Activity) : Plugin(activity) {
    private val worker: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "cubesprite-native").apply { isDaemon = true }
    }
    private var service: CubeSpriteService? = null
    private var insetHost: View? = null

    override fun load(webView: WebView) {
        activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        activity.runOnUiThread {
            WindowCompat.setDecorFitsSystemWindows(activity.window, false)
            enterImmersiveMode()

            val host = activity.findViewById<View>(android.R.id.content) ?: webView
            insetHost = host
            val initialLeft = host.paddingLeft
            val initialTop = host.paddingTop
            val initialRight = host.paddingRight
            val initialBottom = host.paddingBottom
            val safeTypes = WindowInsetsCompat.Type.systemBars() or
                WindowInsetsCompat.Type.displayCutout()

            ViewCompat.setOnApplyWindowInsetsListener(host) { view, windowInsets ->
                val safeInsets = windowInsets.getInsets(safeTypes)
                view.setPadding(
                    initialLeft + safeInsets.left,
                    initialTop + safeInsets.top,
                    initialRight + safeInsets.right,
                    initialBottom + safeInsets.bottom,
                )

                WindowInsetsCompat.Builder(windowInsets)
                    .setInsets(safeTypes, Insets.NONE)
                    .build()
            }
            ViewCompat.requestApplyInsets(host)
        }
    }

    override fun onResume() {
        activity.runOnUiThread {
            enterImmersiveMode()
            insetHost?.let { ViewCompat.requestApplyInsets(it) }
        }
    }

    @Command
    fun request(invoke: Invoke) {
        val args = try {
            invoke.parseArgs(RequestArgs::class.java)
        } catch (error: Exception) {
            invoke.resolveObject(
                errorEnvelope(
                    "INVALID_PARAMS",
                    error.message ?: "Unable to decode Android backend request.",
                    error.javaClass.simpleName,
                ),
            )
            return
        }
        worker.execute {
            try {
                val backend = service ?: CubeSpriteService(activity.applicationContext).also {
                    service = it
                }
                invoke.resolveObject(
                    mapOf(
                        "ok" to true,
                        "result" to backend.handle(args.command, args.params),
                    ),
                )
            } catch (error: ServiceError) {
                invoke.resolveObject(errorEnvelope(error.code, error.message, error.details))
            } catch (error: Exception) {
                invoke.resolveObject(
                    errorEnvelope(
                        "INTERNAL_ERROR",
                        error.message ?: "Unexpected Android backend failure.",
                        error.javaClass.simpleName,
                    ),
                )
            }
        }
    }

    @Command
    fun close(invoke: Invoke) {
        worker.execute {
            try {
                service?.close()
                service = null
                invoke.resolveObject(mapOf("ok" to true, "result" to mapOf("closed" to true)))
            } catch (error: Exception) {
                invoke.resolveObject(
                    errorEnvelope(
                        "INTERNAL_ERROR",
                        error.message ?: "Unable to close Android backend.",
                        error.javaClass.simpleName,
                    ),
                )
            }
        }
    }

    override fun onDestroy(activity: AppCompatActivity) {
        worker.execute {
            service?.close()
            service = null
        }
        worker.shutdown()
    }

    private fun enterImmersiveMode() {
        WindowCompat.getInsetsController(
            activity.window,
            activity.window.decorView,
        ).apply {
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = false
            hide(WindowInsetsCompat.Type.systemBars())
        }
    }

    private fun errorEnvelope(code: String, message: String, details: Any?): Map<String, Any?> =
        mapOf(
            "ok" to false,
            "error" to mapOf(
                "code" to code,
                "message" to message,
                "details" to details,
            ),
        )
}
