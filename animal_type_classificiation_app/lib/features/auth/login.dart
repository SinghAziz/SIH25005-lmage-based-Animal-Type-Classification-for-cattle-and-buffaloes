import 'package:animal_type_classificiation_app/config/app_routes.dart';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../config/app_theme.dart';
import 'signup.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController phoneController = TextEditingController();
  final TextEditingController otpController = TextEditingController();

  final FirebaseAuth _auth = FirebaseAuth.instance;
  String? _verificationId;
  bool _otpSent = false;
  bool _isLoading = false;

  Future<void> _sendOtp() async {
    final phone = phoneController.text.trim();
    if (phone.isEmpty) return;

    setState(() => _isLoading = true);

    await _auth.verifyPhoneNumber(
      phoneNumber: phone,
      verificationCompleted: (PhoneAuthCredential credential) async {
        // Auto verification (for some devices)
        await _auth.signInWithCredential(credential);
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text("Login Successful!")));
      },
      verificationFailed: (FirebaseAuthException e) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Verification failed: ${e.message}")),
        );
      },
      codeSent: (String verificationId, int? resendToken) {
        setState(() {
          _otpSent = true;
          _verificationId = verificationId;
          _isLoading = false;
        });
      },
      codeAutoRetrievalTimeout: (String verificationId) {
        _verificationId = verificationId;
      },
    );
  }

  Future<void> _verifyOtp() async {
    final otp = otpController.text.trim();
    if (_verificationId == null || otp.isEmpty) return;

    setState(() => _isLoading = true);

    try {
      PhoneAuthCredential credential = PhoneAuthProvider.credential(
        verificationId: _verificationId!,
        smsCode: otp,
      );

      await _auth.signInWithCredential(credential);

      setState(() => _isLoading = false);

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text("Login Successful!")));

      Navigator.pushNamed(context, AppRoutes.homepage);
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Invalid OTP: $e")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
      appBar: AppBar(
        backgroundColor: AppTheme.secondaryColor,
        title: Text(
          "Login",
          style: AppTheme.defaultTextStyle(20, fontWeight: FontWeight.w600),
        ),
        centerTitle: true,
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Phone number
              TextField(
                controller: phoneController,
                keyboardType: TextInputType.phone,
                style: AppTheme.defaultTextStyle(16),
                decoration: InputDecoration(
                  labelText: "Phone Number (+91xxxx)",
                  labelStyle: AppTheme.defaultTextStyle(14),
                  filled: true,
                  fillColor: Colors.white,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // OTP field (only if sent)
              if (_otpSent)
                TextField(
                  controller: otpController,
                  keyboardType: TextInputType.number,
                  style: AppTheme.defaultTextStyle(16),
                  decoration: InputDecoration(
                    labelText: "OTP",
                    labelStyle: AppTheme.defaultTextStyle(14),
                    filled: true,
                    fillColor: Colors.white,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              if (_otpSent) const SizedBox(height: 24),

              // Send OTP / Verify OTP Button
              _isLoading
                  ? const CircularProgressIndicator()
                  : ElevatedButton(
                      onPressed: _otpSent ? _verifyOtp : _sendOtp,
                      style: AppTheme.elevatedButtonStyle,
                      child: Text(_otpSent ? "Verify OTP" : "Send OTP"),
                    ),

              const SizedBox(height: 20),

              // Forgot Password
              TextButton(
                onPressed: () {
                  // TODO: Implement forgot password (optional for phone)
                },
                child: Text(
                  "Forgot Password?",
                  style: AppTheme.defaultTextStyle(
                    14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),

              const SizedBox(height: 20),

              // Create Account
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "Don’t have an account?",
                    style: AppTheme.defaultTextStyle(14),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SignupPage(),
                        ),
                      );
                    },
                    child: Text(
                      "Create Account",
                      style: AppTheme.defaultTextStyle(
                        14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
