import 'package:animal_type_classificiation_app/config/app_theme.dart';
import 'package:animal_type_classificiation_app/core/predict.dart';
import 'package:animal_type_classificiation_app/features/prediction/predict_breed.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class HomeContent extends StatefulWidget {
  const HomeContent({super.key});

  @override
  State<HomeContent> createState() => _HomeContentState();
}

class _HomeContentState extends State<HomeContent> {
  File? image;
  final picker = ImagePicker();
  Map<String, dynamic>? prediction;
  bool isLoading = false;

  Future<void> pickImage(ImageSource source) async {
    try {
      final pickedFile = await picker.pickImage(source: source);

      if (pickedFile != null) {
        setState(() {
          image = File(pickedFile.path);
          prediction = null; // Reset prediction for new image
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Error: $e")));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // SizedBox(height: 50),
            Row(
              children: [
                Image.asset(
                  'assets/images/buffalo.png',
                  height: 80,
                  width: 80,
                  fit: BoxFit.cover,
                ),
                Text(
                  "Welcome Back Chief!",
                  style: AppTheme.defaultTextStyle(
                    24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
            const SizedBox(height: 30),

            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: AppTheme.secondaryColor.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: image != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(image!, fit: BoxFit.cover),
                    )
                  : const Center(child: Text("No image selected")),
            ),

            const SizedBox(height: 20),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 120,
                  child: ElevatedButton(
                    onPressed: () => pickImage(ImageSource.camera),
                    style: AppTheme.elevatedButtonStyle,
                    child: const Text("Capture"),
                  ),
                ),
                const SizedBox(width: 16),
                SizedBox(
                  width: 120,
                  child: ElevatedButton(
                    onPressed: () => pickImage(ImageSource.gallery),
                    style: AppTheme.elevatedButtonStyle,
                    child: const Text("Upload"),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: image != null && !isLoading && prediction == null
                  ? () async {
                      setState(() {
                        isLoading = true;
                      });

                      try {
                        final result = await predictImage(image!);

                        setState(() {
                          isLoading = false;
                          prediction = result;
                        });

                        // Auto-navigate to results page
                        if (mounted && result != null) {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => PredictBreed(
                                aiResult: result,
                                imagePath: image!.path,
                              ),
                            ),
                          );
                        }
                      } catch (e) {
                        setState(() {
                          isLoading = false;
                        });
                        if (mounted) {
                          ScaffoldMessenger.of(
                            context,
                          ).showSnackBar(SnackBar(content: Text("Error: $e")));
                        }
                      }
                    }
                  : null,
              style: image != null
                  ? AppTheme.elevatedButtonStyle.copyWith(
                      backgroundColor: const WidgetStatePropertyAll<Color>(
                        Colors.green,
                      ),
                    )
                  : AppTheme.elevatedButtonStyle.copyWith(
                      backgroundColor: WidgetStatePropertyAll(Colors.grey),
                    ),
              child: isLoading
                  ? const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        ),
                        SizedBox(width: 8),
                        Text(
                          "Processing...",
                          style: TextStyle(color: Colors.white),
                        ),
                      ],
                    )
                  : const Text(
                      "Analyze Image",
                      style: TextStyle(color: Colors.white),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
