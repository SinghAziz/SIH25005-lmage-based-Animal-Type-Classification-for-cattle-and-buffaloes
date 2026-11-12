import 'package:animal_type_classificiation_app/core/home_page.dart';
import 'package:animal_type_classificiation_app/features/auth/login.dart';
import 'package:animal_type_classificiation_app/features/auth/signup.dart';
import 'package:animal_type_classificiation_app/features/history/history_page.dart';
import 'package:animal_type_classificiation_app/features/home/home_content.dart';
import 'package:animal_type_classificiation_app/features/map/cattle_map_page.dart';
import 'package:animal_type_classificiation_app/features/prediction/predict_breed.dart';
import 'package:animal_type_classificiation_app/features/settings/settings_page.dart';
import 'package:flutter/material.dart';
import 'features/splash/splash_screen.dart';
import 'config/app_theme.dart';
import 'config/app_routes.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
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
          headlineLarge: AppTheme.defaultTextStyle(
            28,
            fontWeight: FontWeight.bold,
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: AppTheme.elevatedButtonStyle,
        ),
      ),
      initialRoute: AppRoutes.splash,
      routes: {
        AppRoutes.splash: (context) => const SplashScreen(),
        AppRoutes.login: (context) => const LoginPage(),
        AppRoutes.homepage: (context) => const HomePage(),
        AppRoutes.history: (context) => const HistoryPage(),
        AppRoutes.settings: (context) => const SettingsPage(),
        AppRoutes.content: (context) => const HomeContent(),
        AppRoutes.signup: (context) => const SignupPage(),
        AppRoutes.cattleMap: (context) => const CattleMapPage(),
        AppRoutes.predict: (context) {
          final args = ModalRoute.of(context)!.settings.arguments as Map;
          return PredictBreed(
            aiResult: args['aiResult'],
            imagePath: args['imagePath'],
          );
        },
      },
    );
  }
}
