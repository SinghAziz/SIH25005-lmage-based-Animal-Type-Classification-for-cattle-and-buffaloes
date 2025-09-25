import 'package:animal_type_classificiation_app/config/app_theme.dart';
import 'package:animal_type_classificiation_app/features/history/history_page.dart';
import 'package:animal_type_classificiation_app/features/home/home_content.dart';
import 'package:animal_type_classificiation_app/features/settings/settings_page.dart';
import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;
  final List<Widget> _pages = [
    HomeContent(),
    HistoryPage(), 
    SettingsPage(), 
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Cattle Lens",
          style: AppTheme.defaultTextStyle(
            20,
            fontWeight: FontWeight.bold,
          ).copyWith(color: AppTheme.primaryColor),
        ),
        backgroundColor: AppTheme.textColor,
      ),
      body: _pages[_currentIndex],

      // Bottom Navigation Bar
      bottomNavigationBar: BottomNavigationBar(
        selectedItemColor: AppTheme.primaryColor,
        unselectedItemColor: AppTheme.secondaryColor,
        backgroundColor: AppTheme.textColor,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },

        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: "Home"),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: "History"),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: "Settings",
          ),
        ],
      ),
    );
  }
}
