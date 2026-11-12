import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  static const Color primaryColor = Color(0xFFFFF8E1);
  static const Color secondaryColor = Color(0xFF8D6E63);
  static const Color accentColor = Color(0xFF5D4037);
  static const Color textColor = Color(0xFF3E2723);

  static TextStyle defaultTextStyle(
    double fontSize, {
    FontWeight fontWeight = FontWeight.normal,
  }) {
    return GoogleFonts.notoSans(
      fontSize: fontSize,
      fontWeight: fontWeight,
      color: textColor,
    );
  }

  static ButtonStyle elevatedButtonStyle = ElevatedButton.styleFrom(
    backgroundColor: accentColor,
    foregroundColor: Colors.white,
    textStyle: GoogleFonts.notoSans(fontSize: 16, fontWeight: FontWeight.w600),
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
  );
}
