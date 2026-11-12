import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:geolocator/geolocator.dart';
import '../../config/app_theme.dart';

class SimpleCattleListPage extends StatefulWidget {
  const SimpleCattleListPage({super.key});

  @override
  State<SimpleCattleListPage> createState() => _SimpleCattleListPageState();
}

class _SimpleCattleListPageState extends State<SimpleCattleListPage> {
  List<Map<String, dynamic>> _cattleList = [];
  bool _isLoading = true;
  Position? _currentPosition;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    await _getCurrentLocation();
    await _loadCattle();
    setState(() {
      _isLoading = false;
    });
  }

  Future<void> _getCurrentLocation() async {
    try {
      final position = await Geolocator.getCurrentPosition();
      setState(() {
        _currentPosition = position;
      });
      print('📍 Current location: ${position.latitude}, ${position.longitude}');
    } catch (e) {
      print('Location error: $e');
    }
  }

  Future<void> _loadCattle() async {
    try {
      final snapshot = await FirebaseFirestore.instance
          .collection('cattle')
          .get();

      List<Map<String, dynamic>> cattle = [];
      int totalCattle = snapshot.docs.length;
      int nearbyCattle = 0;

      for (var doc in snapshot.docs) {
        final data = doc.data();
        final location = data['location'];

        // Check if cattle has location data
        if (location != null &&
            location['latitude'] != null &&
            location['longitude'] != null &&
            _currentPosition != null) {
          final cattleLat = (location['latitude'] as num).toDouble();
          final cattleLng = (location['longitude'] as num).toDouble();

          // Calculate distance in meters
          double distanceInMeters = Geolocator.distanceBetween(
            _currentPosition!.latitude,
            _currentPosition!.longitude,
            cattleLat,
            cattleLng,
          );

          double distanceInKm = distanceInMeters / 1000;

          // Only include cattle within 3km range
          if (distanceInKm <= 3.0) {
            cattle.add({
              'id': doc.id,
              'tag': data['tag'] ?? 'Unknown',
              'animal': data['animal'] ?? 'Unknown',
              'breed': data['breed'] ?? 'Unknown',
              'location': location,
              'distance': distanceInKm,
            });
            nearbyCattle++;
          }
        }
      }

      // Sort by distance (closest first)
      cattle.sort((a, b) => a['distance'].compareTo(b['distance']));

      setState(() {
        _cattleList = cattle;
      });

      print(
        '🐄 Found $totalCattle total cattle, $nearbyCattle within 3km range',
      );
    } catch (e) {
      print('Error loading cattle: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
      // appBar: AppBar(
      //   title: Text(
      //     'Cattle Locations',
      //     style: AppTheme.defaultTextStyle(
      //       20,
      //       fontWeight: FontWeight.bold,
      //     ).copyWith(color: AppTheme.primaryColor),
      //   ),
      //   backgroundColor: AppTheme.textColor,
      //   iconTheme: const IconThemeData(color: AppTheme.primaryColor),
      // ),
      body: _isLoading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(
                      AppTheme.accentColor,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Loading cattle data...',
                    style: AppTheme.defaultTextStyle(16),
                  ),
                ],
              ),
            )
          : Column(
              children: [
                if (_currentPosition != null)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppTheme.secondaryColor.withOpacity(0.1),
                      border: Border(
                        bottom: BorderSide(
                          color: AppTheme.secondaryColor.withOpacity(0.3),
                          width: 1,
                        ),
                      ),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.my_location,
                              color: AppTheme.accentColor,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Your location: ${_currentPosition!.latitude.toStringAsFixed(4)}, ${_currentPosition!.longitude.toStringAsFixed(4)}',
                                style: AppTheme.defaultTextStyle(
                                  14,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            Icon(
                              Icons.radar,
                              color: AppTheme.accentColor,
                              size: 16,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Showing cattle within 3km radius • ${_cattleList.length} found',
                              style: AppTheme.defaultTextStyle(
                                12,
                                fontWeight: FontWeight.w500,
                              ).copyWith(color: AppTheme.accentColor),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                Expanded(
                  child: _cattleList.isEmpty
                      ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.pets,
                                size: 64,
                                color: AppTheme.secondaryColor,
                              ),
                              const SizedBox(height: 16),
                              Text(
                                'No nearby cattle found',
                                style: AppTheme.defaultTextStyle(
                                  18,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'No cattle within 3km radius',
                                style: AppTheme.defaultTextStyle(
                                  14,
                                ).copyWith(color: AppTheme.secondaryColor),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'Try registering cattle or check a different location',
                                style: AppTheme.defaultTextStyle(
                                  12,
                                ).copyWith(color: AppTheme.secondaryColor),
                              ),
                            ],
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _cattleList.length,
                          itemBuilder: (context, index) {
                            final cattle = _cattleList[index];
                            final location = cattle['location'];

                            return Container(
                              margin: const EdgeInsets.only(bottom: 12),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: [
                                  BoxShadow(
                                    color: AppTheme.secondaryColor.withOpacity(
                                      0.1,
                                    ),
                                    blurRadius: 8,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                                border: Border.all(
                                  color: AppTheme.secondaryColor.withOpacity(
                                    0.2,
                                  ),
                                  width: 1,
                                ),
                              ),
                              child: ListTile(
                                contentPadding: const EdgeInsets.all(16),
                                leading: CircleAvatar(
                                  radius: 25,
                                  backgroundColor: AppTheme.accentColor,
                                  child: Text(
                                    cattle['tag'].length >= 2
                                        ? cattle['tag']
                                              .substring(0, 2)
                                              .toUpperCase()
                                        : cattle['tag'].toUpperCase(),
                                    style: AppTheme.defaultTextStyle(
                                      14,
                                      fontWeight: FontWeight.bold,
                                    ).copyWith(color: Colors.white),
                                  ),
                                ),
                                title: Text(
                                  '🐄 Tag: ${cattle['tag']}',
                                  style: AppTheme.defaultTextStyle(
                                    16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                subtitle: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    const SizedBox(height: 8),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                        vertical: 4,
                                      ),
                                      decoration: BoxDecoration(
                                        color: AppTheme.secondaryColor
                                            .withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(6),
                                      ),
                                      child: Text(
                                        '${cattle['animal']} - ${cattle['breed']}',
                                        style: AppTheme.defaultTextStyle(
                                          12,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                    if (location != null)
                                      Column(
                                        children: [
                                          Row(
                                            children: [
                                              Icon(
                                                Icons.near_me,
                                                size: 14,
                                                color: AppTheme.accentColor,
                                              ),
                                              const SizedBox(width: 4),
                                              Container(
                                                padding:
                                                    const EdgeInsets.symmetric(
                                                      horizontal: 6,
                                                      vertical: 2,
                                                    ),
                                                decoration: BoxDecoration(
                                                  color: AppTheme.accentColor
                                                      .withOpacity(0.1),
                                                  borderRadius:
                                                      BorderRadius.circular(4),
                                                ),
                                                child: Text(
                                                  '${cattle['distance'].toStringAsFixed(2)} km away',
                                                  style:
                                                      AppTheme.defaultTextStyle(
                                                        11,
                                                      ).copyWith(
                                                        color: AppTheme
                                                            .accentColor,
                                                        fontWeight:
                                                            FontWeight.w600,
                                                      ),
                                                ),
                                              ),
                                            ],
                                          ),
                                          const SizedBox(height: 4),
                                          Row(
                                            children: [
                                              Icon(
                                                Icons.location_on,
                                                size: 14,
                                                color: AppTheme.secondaryColor,
                                              ),
                                              const SizedBox(width: 4),
                                              Expanded(
                                                child: Text(
                                                  '${location['latitude']?.toStringAsFixed(4)}, ${location['longitude']?.toStringAsFixed(4)}',
                                                  style:
                                                      AppTheme.defaultTextStyle(
                                                        11,
                                                      ).copyWith(
                                                        color: AppTheme
                                                            .secondaryColor,
                                                      ),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      )
                                    else
                                      Row(
                                        children: [
                                          Icon(
                                            Icons.location_off,
                                            size: 14,
                                            color: AppTheme.secondaryColor,
                                          ),
                                          const SizedBox(width: 4),
                                          Text(
                                            'No location data',
                                            style: AppTheme.defaultTextStyle(12)
                                                .copyWith(
                                                  color:
                                                      AppTheme.secondaryColor,
                                                ),
                                          ),
                                        ],
                                      ),
                                  ],
                                ),
                                trailing: Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: BoxDecoration(
                                    color: location != null
                                        ? AppTheme.accentColor.withOpacity(0.1)
                                        : AppTheme.secondaryColor.withOpacity(
                                            0.1,
                                          ),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Icon(
                                    location != null
                                        ? Icons.location_on
                                        : Icons.location_off,
                                    color: location != null
                                        ? AppTheme.accentColor
                                        : AppTheme.secondaryColor,
                                    size: 20,
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                ),
              ],
            ),
    );
  }
}
