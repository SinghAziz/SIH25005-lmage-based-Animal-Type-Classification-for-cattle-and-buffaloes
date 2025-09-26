  import 'dart:convert';
  import 'dart:io';
  import 'package:http/http.dart' as http;

  Future<Map<String, dynamic>?> predictImage(File imageFile) async {
    var uri = Uri.parse(
      'https://lmage-based-animal-type-classification.onrender.com/predict/',
    );
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
  }
