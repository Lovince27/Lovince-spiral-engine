import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:encrypt/encrypt.dart' as encrypt;

void main() {
  runApp(LovinceAIApp());
}

class LovinceAIApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lovince AI',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: QuantumScreen(),
    );
  }
}

class QuantumScreen extends StatefulWidget {
  @override
  _QuantumScreenState createState() => _QuantumScreenState();
}

class _QuantumScreenState extends State<QuantumScreen> {
  final _userInputController = TextEditingController(text: 'Lovince');
  double _phase = -45.0;
  String _result = 'Lovince AI is ready to inspire!';
  bool _isLoading = false;

  // Secure API communication
  final _encrypter = encrypt.Encrypter(encrypt.AES(encrypt.Key.fromLength(32)));
  final _iv = encrypt.IV.fromLength(16);

  Future<void> _runQuantumCircuit() async {
    setState(() => _isLoading = true);
    try {
      // Encrypt request data
      final encryptedData = _encrypter.encrypt(
        jsonEncode({'user_input': _userInputController.text, 'phase': _phase}),
        iv: _iv,
      );
      
      // Send request to Flask backend
      final response = await http.post(
        Uri.parse('http://your-server:5000/run_quantum'), // Replace with your server
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'data': encryptedData.base64}),
      );

      if (response.statusCode == 200) {
        final decryptedResponse = _encrypter.decrypt64(
          jsonDecode(response.body)['result'],
          iv: _iv,
        );
        setState(() {
          _result = decryptedResponse;
          _isLoading = false;
        });
      } else {
        throw Exception('API request failed');
      }
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Lovince AI: Quantum Living Energy'),
        backgroundColor: Colors.blueAccent,
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _userInputController,
                decoration: InputDecoration(
                  labelText: 'Your Name or Inspiration',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 16),
              Text('Phase Angle (°): ${_phase.toStringAsFixed(1)}'),
              Slider(
                value: _phase,
                min: -180,
                max: 180,
                divisions: 360,
                label: _phase.round().toString(),
                onChanged: (value) => setState(() => _phase = value),
              ),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _isLoading ? null : _runQuantumCircuit,
                child: _isLoading
                    ? CircularProgressIndicator(color: Colors.white)
                    : Text('Run Quantum Circuit'),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 16),
                ),
              ),
              SizedBox(height: 16),
              Text(
                _result,
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              // TODO: Add Bloch sphere and histogram visualizations (e.g., via WebView)
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _userInputController.dispose();
    super.dispose();
  }
}