import 'package:animal_type_classificiation_app/config/app_theme.dart';
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
  String? prediction;
  bool isLoading = false;

  // image picker method

  Future<void> pickImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        image = File(pickedFile.path);
        prediction = null;
      });
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
            Text(
              "Welcome Back Chief!",
              style: AppTheme.defaultTextStyle(24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 30),

            Container(
              height: 250,
              width: double.infinity,
              color: AppTheme.secondaryColor.withOpacity(0.2),
              child: const Center(child: Text("Image will appear here")),
            ),

            const SizedBox(height: 20),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 120, // Adjust the width as needed
                  child: ElevatedButton(
                    onPressed: () {},
                    style: AppTheme.elevatedButtonStyle,
                    child: const Text("Capture"),
                  ),
                ),
                const SizedBox(width: 16),
                SizedBox(
                  width: 120, // Adjust the width as needed
                  child: ElevatedButton(
                    onPressed: () {},
                    style: AppTheme.elevatedButtonStyle,
                    child: const Text("Upload"),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 30),

            // Placeholder for the image
          ],
        ),
      ),

      // Bottom Navigation Bar
     
    );
  }
}
