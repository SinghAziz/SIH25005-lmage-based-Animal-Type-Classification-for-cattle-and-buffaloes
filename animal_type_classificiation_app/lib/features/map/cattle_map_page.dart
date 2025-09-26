import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:geolocator/geolocator.dart';

class CattleMapPage extends StatefulWidget {
  const CattleMapPage({super.key});

  @override
  State<CattleMapPage> createState() => _CattleMapPageState();
}

class _CattleMapPageState extends State<CattleMapPage> {
  Set<Marker> _markers = {};
  LatLng _currentLocation = const LatLng(20.5937, 78.9629);
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _initializeMap();
  }

  Future<void> _initializeMap() async {
    print('🗺️ Starting map initialization...');
    await _getCurrentLocation();
    await _loadCattle();
    print('🗺️ Map initialization complete');
    setState(() {
      _isLoading = false;
    });
  }

  Future<void> _getCurrentLocation() async {
    try {
      final position = await Geolocator.getCurrentPosition();
      setState(() {
        _currentLocation = LatLng(position.latitude, position.longitude);
      });
    } catch (e) {
      print('Location error: $e');
    }
  }

  Future<void> _loadCattle() async {
    try {
      final snapshot = await FirebaseFirestore.instance
          .collection('cattle')
          .get();

      Set<Marker> markers = {};

      for (var doc in snapshot.docs) {
        final data = doc.data();
        final location = data['location'];

        if (location != null && location['latitude'] != null) {
          final lat = (location['latitude'] as num).toDouble();
          final lng = (location['longitude'] as num).toDouble();
          final tag = data['tag'] ?? 'Unknown';

          markers.add(
            Marker(
              markerId: MarkerId(doc.id),
              position: LatLng(lat, lng),
              infoWindow: InfoWindow(title: tag),
            ),
          );
        }
      }

      setState(() {
        _markers = markers;
      });
    } catch (e) {
      print('Error loading cattle: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cattle Map'),
        backgroundColor: const Color(0xFF4CAF50),
      ),
      body: _isLoading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Loading map...'),
                ],
              ),
            )
          : Container(
              color: Colors.grey[300],
              child: GoogleMap(
                onMapCreated: (GoogleMapController controller) {
                  print('✅ Map created successfully!');
                },
                initialCameraPosition: CameraPosition(
                  target: _currentLocation,
                  zoom: 14,
                ),
                markers: _markers,
                myLocationEnabled: true,
                onTap: (LatLng position) {
                  print(
                    'Map tapped at: ${position.latitude}, ${position.longitude}',
                  );
                },
              ),
            ),
    );
  }
}
