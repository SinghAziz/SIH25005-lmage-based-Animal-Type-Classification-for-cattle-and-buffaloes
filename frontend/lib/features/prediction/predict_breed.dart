import 'dart:io';
import 'package:flutter/material.dart';
import 'package:animal_type_classificiation_app/config/app_theme.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:geolocator/geolocator.dart';

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
  // Controllers
  final TextEditingController _tagController = TextEditingController();

  // Location
  Position? _currentPosition;

  @override
  void dispose() {
    _tagController.dispose();
    super.dispose();
  }

  /// Get current location
  Future<void> _fetchLocation() async {
    try {
      print('🌍 Starting location fetch...');

      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      print('🌍 Location service enabled: $serviceEnabled');

      if (!serviceEnabled) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("Location services are disabled. Please enable GPS."),
          ),
        );
        return;
      }

      LocationPermission permission = await Geolocator.checkPermission();
      print('🌍 Current permission: $permission');

      if (permission == LocationPermission.denied) {
        print('🌍 Requesting location permission...');
        permission = await Geolocator.requestPermission();
        print('🌍 Permission after request: $permission');

        if (permission == LocationPermission.denied) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text(
                "Location permission denied. Please grant permission in settings.",
              ),
            ),
          );
          return;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              "Location permissions are permanently denied. Enable them in settings.",
            ),
          ),
        );
        return;
      }

      print('🌍 Getting current position...');
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 10),
      );

      print(
        '🌍 Position received: ${position.latitude}, ${position.longitude}',
      );
      setState(() {
        _currentPosition = position;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            "✅ Location fetched: (${position.latitude.toStringAsFixed(4)}, ${position.longitude.toStringAsFixed(4)})",
          ),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      print('❌ Location error: $e');
      String errorMessage = "Error fetching location: ";

      if (e.toString().contains('MissingPluginException')) {
        errorMessage +=
            "Plugin not properly installed. Try restarting the app.";
      } else if (e.toString().contains('timeout')) {
        errorMessage += "Location request timed out. Try again.";
      } else if (e.toString().contains('permission')) {
        errorMessage += "Location permission required.";
      } else {
        errorMessage += e.toString();
      }

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(errorMessage), backgroundColor: Colors.red),
      );
    }
  }

  /// Save to Firestore
  Future<void> _saveToFirestore() async {
    try {
      final tag = _tagController.text.trim();
      if (tag.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Please enter a Tag number")),
        );
        return;
      }

      final docRef = FirebaseFirestore.instance.collection('cattle').doc(tag);

      // Create a map with all AI result data plus additional fields
      Map<String, dynamic> dataToSave = Map.from(widget.aiResult);

      // Add our custom fields
      dataToSave.addAll({
        'tag': tag,
        'location': _currentPosition != null
            ? {
                'latitude': _currentPosition!.latitude,
                'longitude': _currentPosition!.longitude,
              }
            : null,
        'timestamp': FieldValue.serverTimestamp(),
      });

      await docRef.set(dataToSave);

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text("Data saved successfully!")));
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Error saving data: $e")));
    }
  }

  /// Helper for static fields
  Widget _buildField(String label, dynamic value, {IconData? icon}) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(
            icon ?? _getIconForField(label.toLowerCase()),
            color: AppTheme.primaryColor,
            size: 20,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: AppTheme.defaultTextStyle(
                    14,
                    fontWeight: FontWeight.w500,
                  ).copyWith(color: AppTheme.primaryColor.withOpacity(0.8)),
                ),
                const SizedBox(height: 2),
                Text(
                  "${_formatValue(value)}",
                  style: AppTheme.defaultTextStyle(
                    16,
                    fontWeight: FontWeight.w600,
                  ).copyWith(color: AppTheme.primaryColor),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Get appropriate icon for field
  IconData _getIconForField(String fieldName) {
    switch (fieldName.toLowerCase()) {
      case 'animal':
        return Icons.pets;
      case 'breed':
        return Icons.category;
      case 'atc_score':
      case 'atc score':
        return Icons.score;
      case 'milk_qty':
      case 'milk quantity':
      case 'milk':
        return Icons.water_drop;
      case 'height':
        return Icons.height;
      case 'weight':
        return Icons.monitor_weight;
      case 'age':
        return Icons.calendar_today;
      case 'health_status':
      case 'health status':
      case 'health':
        return Icons.health_and_safety;
      case 'body_condition_score':
      case 'body condition':
        return Icons.fitness_center;
      case 'prediction_confidence':
      case 'confidence':
        return Icons.verified;
      default:
        return Icons.info;
    }
  }

  /// Format value for display
  String _formatValue(dynamic value) {
    if (value == null) return 'Unknown';
    if (value is double) {
      return value.toStringAsFixed(2);
    }
    if (value is num) {
      return value.toString();
    }
    return value.toString();
  }

  /// Format field label for better display
  String _formatFieldLabel(String key) {
    // Convert snake_case and camelCase to Title Case
    String formatted = key
        .replaceAllMapped(RegExp(r'[_-]'), (match) => ' ')
        .replaceAllMapped(
          RegExp(r'([a-z])([A-Z])'),
          (match) => '${match.group(1)} ${match.group(2)}',
        );

    // Capitalize first letter of each word
    return formatted
        .split(' ')
        .map((word) {
          if (word.isEmpty) return word;
          return word[0].toUpperCase() + word.substring(1).toLowerCase();
        })
        .join(' ');
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
            onPressed: _saveToFirestore,
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
                          // Display all prediction fields dynamically
                          ...widget.aiResult.entries.map((entry) {
                            String key = entry.key;
                            dynamic value = entry.value;

                            // Format the field label for better display
                            String label = _formatFieldLabel(key);

                            return Column(
                              children: [
                                _buildField(label, value),
                                if (widget.aiResult.entries.last.key != key)
                                  Divider(
                                    color: AppTheme.primaryColor.withOpacity(
                                      0.3,
                                    ),
                                    thickness: 0.5,
                                  ),
                              ],
                            );
                          }).toList(),

                          const SizedBox(height: 16),

                          // Tag input
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
                                  decoration: InputDecoration(
                                    labelText: "Enter Tag Number",
                                    labelStyle: TextStyle(
                                      color: AppTheme.primaryColor,
                                    ),
                                    border: OutlineInputBorder(
                                      borderSide: BorderSide(
                                        color: AppTheme.primaryColor,
                                      ),
                                    ),
                                    focusedBorder: OutlineInputBorder(
                                      borderSide: BorderSide(
                                        color: AppTheme.primaryColor,
                                      ),
                                    ),
                                    enabledBorder: OutlineInputBorder(
                                      borderSide: BorderSide(
                                        color: AppTheme.primaryColor,
                                      ),
                                    ),
                                  ),
                                  style: TextStyle(
                                    color: AppTheme.primaryColor,
                                  ),
                                  cursorColor: AppTheme.primaryColor,
                                ),
                              ),
                            ],
                          ),

                          const SizedBox(height: 16),

                          // Fetch Location Button
                          ElevatedButton.icon(
                            onPressed: _fetchLocation,
                            icon: const Icon(Icons.my_location),
                            label: Text(
                              _currentPosition == null
                                  ? "Fetch Location"
                                  : "Location Fetched",
                            ),
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
