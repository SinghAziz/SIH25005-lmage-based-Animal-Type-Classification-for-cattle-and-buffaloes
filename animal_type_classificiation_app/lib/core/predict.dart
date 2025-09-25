import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>?> predictImage(File imageFile) async {
  try {
    String baseUrl;

    if (Platform.isIOS) {
      baseUrl = 'http://localhost:8000';
    } else if (Platform.isAndroid) {
      bool isEmulator = true; // Set to false when testing on real device

      if (isEmulator) {
        baseUrl = 'http://10.0.2.2:8000'; // Android Emulator
      } else {
        baseUrl = 'http://192.168.1.100:8000'; // Physical Device (replace IP)
      }
    } else {
      baseUrl = 'http://localhost:8000'; // Desktop/Other
    }

    var uri = Uri.parse('$baseUrl/predict/');

    var request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    var response = await request.send();

    if (response.statusCode == 200) {
      final responseString = await response.stream.bytesToString();
      return responseString.isNotEmpty
          ? Map<String, dynamic>.from(jsonDecode(responseString))
          : null;
    }

    return null;
  } catch (e) {
    print("❌ Error: $e");
    return null;
  }
}
