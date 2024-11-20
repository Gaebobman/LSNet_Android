/*
Author: standard_lee@inha.edu
Date: 2024-11-20
Version: 1.0

 The Test datasets and model file must be placed in the assets folder.
        Structure
        android/app/src/main/assets/
        ├── VT821
        │   ├── RGB
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   └── T
        │       ├── image1.jpg
        │       ├── image2.jpg
        │       └── ...
        ├── VT1000
        │   ├── RGB
        │   └── T
        ├── VT5000
        │   ├── RGB
        │   └── T
        ├── YOUR_ANDROID_MODEL_file.pt

 */


package hci.standardlee.lsnetandroid

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

// TODO: Replace 'aten::empty_strided' Operations in Original Model
class ModelInference(private val context: Context) {

    companion object {
        const val OPTION_RESIZE_AND_PAD = 1
        const val OPTION_SKIP_MISMATCH = 2
        const val IMAGE_SIZE = 224
    }

    private val TAG = "LSNetInference"
    private lateinit var model: Module
    private var totalInferenceTime: Long = 0
    private var processedImageCount: Int = 0

    fun loadModel() {
        // Specify the model path. The model file must be placed in the assets folder.
        // ex) android/app/src/main/assets/{YOUR_ANDROID_MODEL_file.pt}
        val modelPath = "lsnetTorchScript.pt"
        model = Module.load(assetFilePath(modelPath))
    }

    fun runInference(option: Int) {
        // Initialize the counters
        totalInferenceTime = 0
        processedImageCount = 0

        // Process test datasets
        val testDatasets = listOf("VT821", "VT1000", "VT5000")

        testDatasets.forEach { dataset ->
            Log.d(TAG, "Current Dataset: $dataset")
            val rgbRoot = "$dataset/RGB"
            val tiRoot = "$dataset/T"
            val rgbs = getFilesFromAssets(rgbRoot)
            val tis = getFilesFromAssets(tiRoot)

            rgbs.zip(tis).forEach { (rgbPath, tiPath) ->
                val rgbBitmap = BitmapFactory.decodeStream(context.assets.open(rgbPath))
                val tiBitmap = BitmapFactory.decodeStream(context.assets.open(tiPath))

                // Perform inference and measure time
                val startTime = System.currentTimeMillis()
                processAndInfer(rgbBitmap, tiBitmap, "$dataset/${rgbPath.substringAfterLast("/").removeSuffix(".jpg")}")
                val endTime = System.currentTimeMillis()

                // Accumulate inference time and count processed images
                totalInferenceTime += (endTime - startTime)
                processedImageCount++
            }
        }

        // Calculate and log the average inference time after all images are processed
        if (processedImageCount > 0) {
            val averageInferenceTime = totalInferenceTime.toDouble() / processedImageCount
            Log.d(TAG, "Processed $processedImageCount images")
            Log.d(TAG, "Average Inference Time: $averageInferenceTime ms")
        }
    }
    private fun processAndInfer(rgbBitmap: Bitmap, tiBitmap: Bitmap, filePath: String) {
        // Resize images to match the model input size
        val resizedRgbBitmap = Bitmap.createScaledBitmap(rgbBitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val resizedTiBitmap = Bitmap.createScaledBitmap(tiBitmap, IMAGE_SIZE, IMAGE_SIZE, true)

        // Convert the images to tensors
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedRgbBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val tiTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedTiBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // Run the model inference
        val startTime = System.currentTimeMillis()
        val output = model.forward(IValue.from(inputTensor), IValue.from(tiTensor))
        val endTime = System.currentTimeMillis()
        Log.d(TAG, "InferenceTime: ${endTime - startTime} ms")

        // Handle model output
        val outputTensor: Tensor = when (output) {
            is IValue -> {
                if (output.isTuple) {
                    output.toTuple()[0].toTensor() // Extract the first element if the output is a tuple
                } else {
                    output.toTensor() // Handle a single tensor output
                }
            }
            else -> throw IllegalStateException("Unexpected model output type")
        }

        val scores = outputTensor.dataAsFloatArray

        // Process and save the result as a Bitmap
        val resultBitmap = processResult(scores)
        Log.d(TAG, "InferenceResult: $resultBitmap")
        saveBitmap(resultBitmap, "results/$filePath.png")
    }

    private fun grayScale(orgBitmap: Bitmap): Bitmap {
        val width = orgBitmap.width
        val height = orgBitmap.height

        // Create a new grayscale Bitmap with ARGB_8888 configuration
        val bmpGrayScale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Create a Canvas to draw the grayscale Bitmap
        val canvas = Canvas(bmpGrayScale)
        val paint = Paint()

        // Set a ColorMatrix to convert to grayscale
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f) // Set saturation to 0 to create a grayscale effect
        val colorMatrixFilter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = colorMatrixFilter

        // Draw the original Bitmap onto the canvas using the grayscale paint
        canvas.drawBitmap(orgBitmap, 0f, 0f, paint)

        return bmpGrayScale
    }

    private fun processResult(scores: FloatArray): Bitmap {
        // Create a Bitmap to store the output
        val bitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(scores.size)

        // Scale the scores array values to [0, 255] and set them as pixel values
        for (i in scores.indices) {
            // Assume scores array values are in the range [0, 1], convert them to [0, 255] grayscale
            val gray = (scores[i] * 255).toInt().coerceIn(0, 255) // Limit values to 0-255
            // Convert to ARGB format
            pixels[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        }

        // Set the pixel data into the Bitmap
        bitmap.setPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Convert the Bitmap to grayscale
        return grayScale(bitmap)
    }

    private fun saveBitmap(bitmap: Bitmap, filePath: String) {
        try {
            // Create the file and save the Bitmap as a PNG
            val file = File(context.filesDir, filePath).apply {
                parentFile?.mkdirs() // Create parent directories if needed
            }
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // Save as PNG
                Log.d(TAG, "Image saved to ${file.absolutePath}")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error saving image", e)
        }
    }

    private fun assetFilePath(assetName: String): String {
        // Copy the asset file to the internal storage and return its path
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath // Return the existing file if it's already copied
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        return file.absolutePath
    }

    private fun getFilesFromAssets(path: String): List<String> {
        // Retrieve a list of files from the assets folder
        return try {
            context.assets.list(path)?.map { "$path/$it" }
                ?.filter { it.endsWith(".png") || it.endsWith(".jpg") } ?: emptyList()
        } catch (e: IOException) {
            emptyList()
        }
    }
}
