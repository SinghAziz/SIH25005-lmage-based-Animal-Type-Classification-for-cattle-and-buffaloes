import 'package:animal_type_classificiation_app/core/home/home_page.dart';
import 'package:flutter/material.dart';
import 'core/splash/splash_screen.dart';
import 'config/app_theme.dart';
import 'config/app_routes.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cattle Identifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppTheme.primaryColor,
        textTheme: TextTheme(
          bodyLarge: AppTheme.defaultTextStyle(16),
          bodyMedium: AppTheme.defaultTextStyle(14),
          headlineLarge: AppTheme.defaultTextStyle(28, fontWeight: FontWeight.bold),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: AppTheme.elevatedButtonStyle,
        ),
      ),
      initialRoute: AppRoutes.splash,
      routes: {
        AppRoutes.splash: (context) => const SplashScreen(),
        // Uncomment and add other screens later
        // AppRoutes.login: (context) => const LoginScreen(),
        AppRoutes.home: (context) => const HomePage(),
        // AppRoutes.cattle: (context) => const CattleScreen(),
      },
    );
  }
}
