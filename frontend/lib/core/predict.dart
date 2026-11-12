import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart' show XFile;

const apiUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://localhost:8000/predict',
);

Future<Map<String, dynamic>?> predictImage(XFile imageFile) async {
  var uri = Uri.parse(apiUrl);
  var bytes = await imageFile.readAsBytes();
  var request = http.MultipartRequest('POST', uri)
    ..files.add(
      http.MultipartFile.fromBytes('file', bytes, filename: imageFile.name),
    );

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
