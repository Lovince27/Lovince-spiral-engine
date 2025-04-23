import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';
import 'package:cached_network_image/cached_network_image.dart';

void main() {
  runApp(LovinceAIApp());
}

class LovinceAIApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lovince AI',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.teal,
        scaffoldBackgroundColor: Colors.black,
        textTheme: TextTheme(
          bodyLarge: TextStyle(color: Colors.white),
          bodyMedium: TextStyle(color: Colors.grey),
        ),
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final String serverUrl = "http://your_server_ip:5000"; // Replace with your server IP
  Map<String, dynamic>? state;
  AudioPlayer audioPlayer = AudioPlayer();
  bool isPlaying = false;
  String? errorMessage;

  @override
  void initState() {
    super.initState();
    startAI();
    fetchState();
  }

  Future<void> startAI() async {
    try {
      final response = await http.get(Uri.parse("$serverUrl/start"));
      if (response.statusCode != 200) {
        setState(() {
          errorMessage = "Failed to start AI: ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = "Error starting AI: $e";
      });
    }
  }

  Future<void> fetchState() async {
    while (mounted) {
      try {
        final response = await http.get(Uri.parse("$serverUrl/state"));
        if (response.statusCode == 200) {
          setState(() {
            state = jsonDecode(response.body);
            errorMessage = null;
          });
        } else {
          setState(() {
            errorMessage = "Failed to fetch state: ${response.body}";
          });
        }
      } catch (e) {
        setState(() {
          errorMessage = "Error fetching state: $e";
        });
      }
      await Future.delayed(Duration(seconds: 2));
    }
  }

  Future<void> playSound(String soundUrl) async {
    try {
      if (isPlaying) {
        await audioPlayer.stop();
      }
      await audioPlayer.play(UrlSource(soundUrl));
      setState(() {
        isPlaying = true;
      });
    } catch (e) {
      setState(() {
        errorMessage = "Error playing sound: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Lovince AI - The Founder - Lovince ™"),
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: () => fetchState(),
          ),
        ],
      ),
      body: errorMessage != null
          ? Center(child: Text(errorMessage!, style: TextStyle(color: Colors.red)))
          : state == null
              ? Center(child: CircularProgressIndicator())
              : SingleChildScrollView(
                  child: Column(
                    children: [
                      // Cosmic spiral
                      Container(
                        margin: EdgeInsets.all(10),
                        child: CachedNetworkImage(
                          imageUrl: "data:image/png;base64,${state!['plot']}",
                          placeholder: (context, url) => CircularProgressIndicator(),
                          errorWidget: (context, url, error) => Icon(Icons.error, color: Colors.red),
                        ),
                      ),
                      // Play sound button
                      ElevatedButton(
                        onPressed: () => playSound("$serverUrl/sound/${state!['sound_file']}"),
                        child: Text(isPlaying ? "Stop Sound" : "Play Sound"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal,
                          padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                        ),
                      ),
                      // SME insights
                      Padding(
                        padding: EdgeInsets.all(10),
                        child: Text(
                          state!['insights'],
                          style: TextStyle(fontSize: 16),
                        ),
                      ),
                      // Metrics
                      Padding(
                        padding: EdgeInsets.all(10),
                        child: Text(
                          "Frequency: ${state!['frequency'].toStringAsFixed(2)} Hz\n"
                          "Pressure: ${state!['pressure'].toStringAsFixed(2)}\n"
                          "Rhythm: ${state!['rhythm'].toStringAsFixed(2)}",
                          style: TextStyle(fontSize: 14, color: Colors.grey),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }

  @override
  void dispose() {
    audioPlayer.dispose();
    super.dispose();
  }
}