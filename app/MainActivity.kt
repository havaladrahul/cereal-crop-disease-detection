package com.example.crop_disease_detector

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.crop_disease_detector.databinding.ActivityMainBinding
import com.google.android.material.snackbar.Snackbar
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.DecimalFormat
import kotlin.math.exp

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var tflite: Interpreter

    private val diseases = listOf(
        "Corn_Blight",
        "Corn_Common_Rust",
        "Corn_Gray_Leaf_Spot",
        "Corn_Healthy",
        "Rice_Bacterial_Blight",
        "Rice_Blast",
        "Rice_Brown_Spot",
        "Sorghum_Anthracnose",
        "Sorghum_Cereal_Grain_Molds",
        "Sorghum_Head_Smut",
        "Sorghum_Loose_Smut",
        "Sorghum_Rust",
        "Wheat_Brown_Rust",
        "Wheat_Healthy",
        "Wheat_Septoria",
        "Wheat_Yellow_Rust"
    )

    private val inputImageSize = 224 // Your model's expected input size
    private val decimalFormat = DecimalFormat("0.0%")
    private val TAG = "CropDiseaseDetector"

    // Pick image from gallery
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            try {
                val bitmap = contentResolver.openInputStream(it)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
                if (bitmap != null) {
                    binding.imageView.setImageBitmap(bitmap)
                    binding.resultText.text = predictDisease(bitmap)
                } else {
                    showError("Failed to load image")
                }
            } catch (e: Exception) {
                showError("Error loading image: ${e.message}")
                Log.e(TAG, "Image loading error", e)
            }
        }
    }

    // Take picture using camera
    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        if (bitmap != null) {
            binding.imageView.setImageBitmap(bitmap)
            binding.resultText.text = predictDisease(bitmap)
        } else {
            showError("Failed to capture image")
        }
    }

    // Request camera permission
    private val requestCameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            takePicture.launch(null)
        } else {
            showError("Camera permission denied")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        // Load TFLite model
        try {
            val modelFile = FileUtil.loadMappedFile(this, "mobilenet_v2_crop_disease.tflite")
            tflite = Interpreter(modelFile)
            Log.i(TAG, "Model loaded successfully. Input tensors: ${tflite.inputTensorCount}, Output tensors: ${tflite.outputTensorCount}")
            val inputShape = tflite.getInputTensor(0).shape().joinToString(", ")
            val outputShape = tflite.getOutputTensor(0).shape().joinToString(", ")
            Log.i(TAG, "Input shape: [$inputShape], Output shape: [$outputShape]")
        } catch (e: Exception) {
            showError("Error loading model: ${e.message}")
            Log.e(TAG, "Model loading error", e)
        }

        binding.uploadButton.setOnClickListener {
            pickImage.launch("image/*")
        }

        binding.cameraCaptureButton.setOnClickListener {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun predictDisease(bitmap: Bitmap): String {
        try {
            // Resize bitmap to model input size
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputImageSize, inputImageSize, true)

            // Prepare input buffer (int8 quantized model expects bytes)
            val inputBuffer = ByteBuffer.allocateDirect(inputImageSize * inputImageSize * 3)
            inputBuffer.order(ByteOrder.nativeOrder())
            val pixels = IntArray(inputImageSize * inputImageSize)
            scaledBitmap.getPixels(pixels, 0, inputImageSize, 0, 0, inputImageSize, inputImageSize)

            for (pixel in pixels) {
                val r = (((pixel shr 16) and 0xFF) - 128).toByte()
                val g = (((pixel shr 8) and 0xFF) - 128).toByte()
                val b = ((pixel and 0xFF) - 128).toByte()
                inputBuffer.put(r)
                inputBuffer.put(g)
                inputBuffer.put(b)
            }

            // Prepare output buffer for int8 (size = number of classes)
            val outputBuffer = ByteBuffer.allocateDirect(diseases.size)
            outputBuffer.order(ByteOrder.nativeOrder())

            // Run model inference
            tflite.run(inputBuffer, outputBuffer)

            // Get quantization params for output tensor
            val outputTensor = tflite.getOutputTensor(0)
            val scale = outputTensor.quantizationParams().scale
            val zeroPoint = outputTensor.quantizationParams().zeroPoint

            outputBuffer.rewind()
            val rawScores = FloatArray(diseases.size)
            val rawBytes = ByteArray(diseases.size)
            outputBuffer.get(rawBytes)

            for (i in rawScores.indices) {
                val int8Value = rawBytes[i].toInt()
                // Dequantize output
                rawScores[i] = (int8Value - zeroPoint) * scale
            }

            // Clamp scores to [0, 1]
            val clampedScores = rawScores.map { it.coerceIn(0f, 1f) }

            // Softmax normalization for confidence distribution
            val expScores = clampedScores.map { exp(it.toDouble()) }
            val sumExp = expScores.sum()
            val normalizedScores = expScores.map { (it / sumExp).toFloat() }

            // Find max confidence and index
            val maxIndex = normalizedScores.indices.maxByOrNull { normalizedScores[it] } ?: 0
            val confidence = normalizedScores[maxIndex]

            // Format result string
            return "${diseases[maxIndex]} (${decimalFormat.format(confidence)})"
        } catch (e: Exception) {
            showError("Error predicting disease: ${e.message}")
            Log.e(TAG, "Inference error", e)
            return "Prediction failed"
        }
    }

    private fun showError(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::tflite.isInitialized) {
            tflite.close()
        }
    }
}

