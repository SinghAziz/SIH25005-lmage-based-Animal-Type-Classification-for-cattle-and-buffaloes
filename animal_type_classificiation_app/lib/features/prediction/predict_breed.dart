import 'package:flutter/material.dart';
import 'dart:io';
import 'package:animal_type_classificiation_app/config/app_theme.dart';

class PredictBreed extends StatefulWidget {
  final Map<String, dynamic> aiResult;
  final String imagePath;

  const PredictBreed({
    super.key,
    required this.aiResult,
    required this.imagePath,
  });

  @override
  State<PredictBreed> createState() => _PredictBreedState();
}

class _PredictBreedState extends State<PredictBreed> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor, // ✅ use theme bg
      appBar: AppBar(
        title: Text(
          "Prediction Result",
          style: AppTheme.defaultTextStyle(
            20,
            fontWeight: FontWeight.bold,
          ).copyWith(color: AppTheme.primaryColor),
        ),
        iconTheme: IconThemeData(color: AppTheme.primaryColor),
        backgroundColor: AppTheme.textColor,
        centerTitle: true,
        elevation: 2,

        actions: [
          IconButton(
            onPressed: () {
              // TODO: add history save action
            },
            icon: Icon(Icons.save, color: AppTheme.primaryColor),
            tooltip: "Save to History",
          ),
        ],
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            /// 🖼️ Image Preview Card
            Card(
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              elevation: 4,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.file(
                  File(widget.imagePath),
                  height: 220,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              ),
            ),
            const SizedBox(height: 20),

            Card(
              color: AppTheme.accentColor, 
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Icon(Icons.pets, color: AppTheme.primaryColor),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            "Animal: ${widget.aiResult['animal'] ?? 'Unknown'}",
                            style: AppTheme.defaultTextStyle(
                              20,
                              fontWeight: FontWeight.bold,
                            ).copyWith(color: AppTheme.primaryColor),
                          ),
                        ),
                      ],
                    ),
                    const Divider(height: 24),
                    Row(
                      children: [
                        Icon(Icons.info_outline, color: AppTheme.primaryColor),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            "Breed: ${widget.aiResult['breed'] ?? 'Unknown'}",
                            style: AppTheme.defaultTextStyle(
                              20,
                              fontWeight: FontWeight.bold,
                            ).copyWith(color: AppTheme.primaryColor),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
