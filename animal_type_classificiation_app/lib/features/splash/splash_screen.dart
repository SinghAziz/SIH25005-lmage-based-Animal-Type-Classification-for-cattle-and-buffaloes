import 'package:google_fonts/google_fonts.dart';
import 'package:flutter/material.dart';
import '../../../config/app_theme.dart';
import '../../../config/app_routes.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  final Map<String, String> languages = {
    'en': 'English',
    'hi': 'Hindi',
    'pa': 'Punjabi',
  };

  String selectedLang = 'en';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
      body: Center(
        child: Column(
          // mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Image.asset(
                  'assets/images/buffalo.png',
                  height: 80,
                  width: 80,
                  fit: BoxFit.cover,
                ),
                Text(
                  "Cattle Lens",
                  style: TextStyle(
                    color: AppTheme.textColor,
                    fontSize: 50,
                    fontFamily: GoogleFonts.aBeeZee().fontFamily,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),

            const SizedBox(height: 30),

            ElevatedButton(
              onPressed: () {
                Navigator.pushReplacementNamed(context, AppRoutes.login);
              },
              child: const Text("Continue"),
            ),
          ],
        ),
      ),
    );
  }
}
