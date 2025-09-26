import 'dart:io';
import 'package:flutter/material.dart';
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
  // Controller for Tag input
  final TextEditingController _tagController = TextEditingController();

  @override
  void dispose() {
    _tagController.dispose();
    super.dispose();
  }

  /// Helper to build display fields
  Widget _buildField(String label, dynamic value) {
    return Row(
      children: [
        Icon(Icons.label, color: AppTheme.primaryColor),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            "$label: ${value ?? 'Unknown'}",
            style: AppTheme.defaultTextStyle(
              18,
              fontWeight: FontWeight.w600,
            ).copyWith(color: AppTheme.primaryColor),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
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
              // TODO: Save to Firestore later
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text("Save functionality coming soon!"),
                ),
              );
            },
            icon: Icon(Icons.save, color: AppTheme.primaryColor),
            tooltip: "Save to History",
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  /// 🖼️ Image Preview
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

                  /// Prediction Fields
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
                          _buildField("Animal", widget.aiResult['animal']),
                          const Divider(),
                          _buildField("Breed", widget.aiResult['breed']),
                          const Divider(),
                          _buildField(
                            "ATC Score",
                            widget.aiResult['atc_score'],
                          ),
                          const Divider(),
                          _buildField(
                            "Milk Quantity",
                            widget.aiResult['milk_qty'],
                          ),
                          const Divider(),
                          _buildField("Height", widget.aiResult['height']),
                          const Divider(),
                          _buildField("Weight", widget.aiResult['weight']),
                          const Divider(),

                          // Tag input (User provided)
                          Row(
                            children: [
                              Icon(
                                Icons.confirmation_number,
                                color: AppTheme.primaryColor,
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: TextField(
                                  controller: _tagController,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                    hintText: "Enter Tag Number",
                                    hintStyle: TextStyle(
                                      color: AppTheme.primaryColor,
                                    ),
                                    labelStyle: TextStyle(
                                      color: AppTheme.primaryColor,
                                    ),
                                    enabledBorder: OutlineInputBorder(
                                      borderSide: BorderSide(
                                        color: AppTheme.primaryColor,
                                      ),
                                    ),
                                    focusedBorder: OutlineInputBorder(
                                      borderSide: BorderSide(
                                        color: AppTheme.primaryColor,
                                      ),
                                    ),
                                    border: OutlineInputBorder(),
                                  ),
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
          ),
        ],
      ),
    );
  }
}
