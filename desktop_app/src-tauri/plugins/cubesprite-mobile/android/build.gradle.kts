import org.gradle.api.GradleException
import org.gradle.api.tasks.Copy

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val cubeSpriteResourceRoot = file("../../../resources")
val cubeSpriteModelFiles = listOf(
    cubeSpriteResourceRoot.resolve("models/cubesprite_v3.onnx"),
    cubeSpriteResourceRoot.resolve("models/cubesprite_v3_mini.onnx"),
)
val generatedCubeSpriteAssets = layout.buildDirectory.dir("generated/cubespriteAssets")

val prepareCubeSpriteAssets by tasks.registering(Copy::class) {
    doFirst {
        cubeSpriteModelFiles.forEach { model ->
            if (!model.isFile || model.length() < 1_000_000L) {
                throw GradleException(
                    "CubeSprite model ${model.name} is missing or is still a Git LFS pointer. Run git lfs pull."
                )
            }
        }
    }
    from(cubeSpriteResourceRoot.resolve("model_registry.android.json")) {
        rename { "model_registry.json" }
    }
    from(cubeSpriteResourceRoot.resolve("models")) {
        include("cubesprite_v3.onnx", "cubesprite_v3_mini.onnx")
        into("models")
    }
    into(generatedCubeSpriteAssets)
}

android {
    namespace = "com.cubesprite.mobile"
    compileSdk = 36

    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    sourceSets.getByName("main").assets.srcDir(generatedCubeSpriteAssets)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareCubeSpriteAssets)
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
    implementation(project(":tauri-android"))

    testImplementation("junit:junit:4.13.2")
}
