// lib/features/splash/presentation/splash_screen.dart
import 'package:flutter/material.dart';
import '../../../config/app_theme.dart';
import '../../../config/app_routes.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  // Available languages
  final Map<String, String> languages = {
    'en': 'English',
    'hi': 'Hindi',
    'pa': 'Punjabi',
  };

  String selectedLang = 'en'; // default language

  // Empty function for language change - can be implemented later

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
      body: Center(
        child: Column(
          // mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // App Name
            Text(
              "Cattle Lens",
              style: TextStyle(color: AppTheme.textColor, fontSize: 60),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 30),

            // Continue Button
            ElevatedButton(
              onPressed: () {
                Navigator.pushReplacementNamed(context, AppRoutes.homepage);
              },
              child: const Text("Continue"),
            ),
          ],
        ),
      ),
    );
  }
}
