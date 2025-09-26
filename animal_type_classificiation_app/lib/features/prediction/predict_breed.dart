import 'package:flutter/material.dart';
import 'dart:io';
import 'package:animal_type_classificiation_app/config/app_theme.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';

class PredictBreed extends StatefulWidget {
  final Map<String, dynamic> aiResult;
  final String imagePath;

  const PredictBreed({
    super.key,
    required this.aiResult,
    required this.imagePath,
  });

  @override
  State<PredictBreed> createState() => _PredictBreedState();
}

class _PredictBreedState extends State<PredictBreed> {
  // Speech to text
  late stt.SpeechToText _speech;
  bool _isListening = false;
  String _lastWords = '';

  // Text to speech (kept for the "Read Result Aloud" button)
  final FlutterTts _flutterTts = FlutterTts();

  // Text field controller
  final TextEditingController _textController = TextEditingController();

  // Messages list used for demo (user message + translation)
  final List<Map<String, dynamic>> _messages = [];
  final ScrollController _messagesScrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
  }

  @override
  void dispose() {
    try {
      _speech.stop();
    } catch (_) {}
    _flutterTts.stop();
    _textController.dispose();
    _messagesScrollController.dispose();
    super.dispose();
  }

  /// Simple mock translation - replace with real translator later
  String _mockTranslate(String text) {
    if (text.trim().isEmpty) return "";
    // Placeholder: just annotate as a "Translated" string.
    return "Translated (mock): $text";
  }

  bool _isDuplicateUserMessage(String text) {
    if (text.trim().isEmpty) return true;
    for (int i = _messages.length - 1; i >= 0; i--) {
      if (_messages[i]['type'] == 'user') {
        return _messages[i]['text'] == text;
      }
    }
    return false;
  }

  void _addMessagePair(String userText) {
    final trimmed = userText.trim();
    if (trimmed.isEmpty) return;
    if (_isDuplicateUserMessage(trimmed)) return;

    setState(() {
      _messages.add({'type': 'user', 'text': trimmed, 'time': DateTime.now()});

      _messages.add({
        'type': 'translation',
        'text': _mockTranslate(trimmed),
        'time': DateTime.now(),
      });

      // clear the textfield after adding
      _textController.clear();
    });

    // scroll to bottom
    Future.delayed(const Duration(milliseconds: 120), () {
      if (_messagesScrollController.hasClients) {
        try {
          _messagesScrollController.animateTo(
            _messagesScrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        } catch (_) {}
      }
    });
  }

  /// Toggle listening: start or stop
  void _listen() async {
    if (!_isListening) {
      bool available = await _speech.initialize(
        onStatus: (status) {
          // if recognition finished by itself, add message (guarded)
          // status values vary by platform: 'notListening', 'done', etc
          if (status == 'notListening' || status == 'done') {
            if (_lastWords.trim().isNotEmpty) {
              _addMessagePair(_lastWords);
            }
            setState(() => _isListening = false);
          }
          // useful for debugging:
          // print('Speech status: $status');
        },
        onError: (error) {
          // print('Speech error: $error');
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('Speech error: $error')));
        },
      );

      if (available) {
        setState(() => _isListening = true);
        _speech.listen(
          onResult: (val) {
            // update the textfield live
            setState(() {
              _lastWords = val.recognizedWords;
              _textController.text = _lastWords;
              _textController.selection = TextSelection.fromPosition(
                TextPosition(offset: _textController.text.length),
              );
            });

            // if the package reports a final result, add it
            // (not all platforms may set finalResult reliably; keep the status handler too)
            try {
              if (val.finalResult == true) {
                _addMessagePair(val.recognizedWords);
                setState(() => _isListening = false);
                _speech.stop();
              }
            } catch (e) {
              // ignore if finalResult isn't available
            }
          },
          listenFor: const Duration(seconds: 30),
          pauseFor: const Duration(seconds: 3),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Speech recognition not available")),
        );
      }
    } else {
      // stop listening and add what we have
      try {
        await _speech.stop();
      } catch (_) {}
      setState(() => _isListening = false);
      if (_lastWords.trim().isNotEmpty) {
        _addMessagePair(_lastWords);
      }
    }
  }

  /// Speak AI result (kept from your earlier UI)
  Future<void> _speakResult() async {
    String text =
        "Animal: ${widget.aiResult['animal'] ?? 'Unknown'}, Breed: ${widget.aiResult['breed'] ?? 'Unknown'}";
    await _flutterTts.speak(text);
  }

  Widget _buildMessageTile(Map<String, dynamic> msg) {
    final isUser = msg['type'] == 'user';
    final bubbleColor = isUser ? AppTheme.accentColor : Colors.white;
    final textColor = isUser ? Colors.white : AppTheme.textColor;
    final align = isUser ? Alignment.centerRight : Alignment.centerLeft;
    final radius = isUser
        ? const BorderRadius.only(
            topLeft: Radius.circular(12),
            topRight: Radius.circular(12),
            bottomLeft: Radius.circular(12),
          )
        : const BorderRadius.only(
            topLeft: Radius.circular(12),
            topRight: Radius.circular(12),
            bottomRight: Radius.circular(12),
          );

    return Align(
      alignment: align,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 14),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        decoration: BoxDecoration(
          color: bubbleColor,
          borderRadius: radius,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Text(
          msg['text'] ?? '',
          style: AppTheme.defaultTextStyle(15).copyWith(color: textColor),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryColor,
      appBar: AppBar(
        title: Text(
          "Prediction Result",
          style: AppTheme.defaultTextStyle(
            20,
            fontWeight: FontWeight.bold,
          ).copyWith(color: AppTheme.primaryColor),
        ),
        iconTheme: IconThemeData(color: AppTheme.primaryColor),
        backgroundColor: AppTheme.textColor,
        centerTitle: true,
        elevation: 2,
        actions: [
          IconButton(
            onPressed: () {
              // keep save as-is (no backend change)
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text("Save functionality coming soon!"),
                ),
              );
            },
            icon: Icon(Icons.save, color: AppTheme.primaryColor),
            tooltip: "Save to History",
          ),
        ],
      ),
      body: Column(
        children: [
          // Top: image + result + read button
          Expanded(
            flex: 6,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  // Image Preview
                  Card(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 4,
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: Image.file(
                        File(widget.imagePath),
                        height: 220,
                        width: double.infinity,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),

                  // Prediction Card
                  Card(
                    color: AppTheme.accentColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: [
                          Row(
                            children: [
                              Icon(Icons.pets, color: AppTheme.primaryColor),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  "Animal: ${widget.aiResult['animal'] ?? 'Unknown'}",
                                  style: AppTheme.defaultTextStyle(
                                    20,
                                    fontWeight: FontWeight.bold,
                                  ).copyWith(color: AppTheme.primaryColor),
                                ),
                              ),
                            ],
                          ),
                          const Divider(height: 24),
                          Row(
                            children: [
                              Icon(
                                Icons.info_outline,
                                color: AppTheme.primaryColor,
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  "Breed: ${widget.aiResult['breed'] ?? 'Unknown'}",
                                  style: AppTheme.defaultTextStyle(
                                    20,
                                    fontWeight: FontWeight.bold,
                                  ).copyWith(color: AppTheme.primaryColor),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),

                  // TTS read button
                  ElevatedButton.icon(
                    onPressed: _speakResult,
                    icon: const Icon(Icons.volume_up, color: Colors.white),
                    label: const Text(
                      "Read Result Aloud",
                      style: TextStyle(color: Colors.white),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.accentColor,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 12,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Middle: messages list (user message + translation)
          Expanded(
            flex: 4,
            child: Container(
              color: AppTheme.primaryColor.withOpacity(0.02),
              child: _messages.isEmpty
                  ? Center(
                      child: Text(
                        "No messages yet. Tap the mic and speak to add a message + translation tile.",
                        textAlign: TextAlign.center,
                        style: AppTheme.defaultTextStyle(
                          14,
                        ).copyWith(color: AppTheme.textColor.withOpacity(0.8)),
                      ),
                    )
                  : ListView.builder(
                      controller: _messagesScrollController,
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      itemCount: _messages.length,
                      itemBuilder: (context, idx) {
                        return _buildMessageTile(_messages[idx]);
                      },
                    ),
            ),
          ),

          // Bottom: Input bar with TextField + mic button
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            color: AppTheme.textColor.withOpacity(0.06),
            child: SafeArea(
              top: false,
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _textController,
                      decoration: InputDecoration(
                        hintText: "Type or speak...",
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                        filled: true,
                        fillColor: Colors.white,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 12,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    decoration: BoxDecoration(
                      color: _isListening ? Colors.red : AppTheme.accentColor,
                      shape: BoxShape.circle,
                    ),
                    child: IconButton(
                      icon: Icon(
                        _isListening ? Icons.mic : Icons.mic_none,
                        color: Colors.white,
                      ),
                      onPressed: _listen,
                      tooltip: _isListening
                          ? "Stop listening"
                          : "Start voice input",
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
