import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>?> predictImage(File imageFile) async {
  var uri = Uri.parse('http://172.20.10.8:8000/predict');
  var request = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  var response = await request.send();

  if (response.statusCode == 200) {
    final responseString = await response.stream.bytesToString();
    print(responseString);
    return responseString.isNotEmpty
        ? Map<String, dynamic>.from(jsonDecode(responseString))
        : null;
  }
  return null;
}
