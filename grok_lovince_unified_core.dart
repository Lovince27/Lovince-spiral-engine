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
      theme: ThemeData.dark(),
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

  @override
  void initState() {
    super.initState();
    startAI();
    fetchState();
  }

  Future<void> startAI() async {
    try {
      await http.get(Uri.parse("$serverUrl/start"));
    } catch (e) {
      print("Error starting AI: $e");
    }
  }

  Future<void> fetchState() async {
    while (true) {
      try {
        final response = await http.get(Uri.parse("$serverUrl/state"));
        if (response.statusCode == 200) {
          setState(() {
            state = jsonDecode(response.body);
          });
        }
      } catch (e) {
        print("Error fetching state: $e");
      }
      await Future.delayed(Duration(seconds: 2)); // Poll every 2 seconds
    }
  }

  Future<void> playSound(String soundUrl) async {
    if (isPlaying) {
      await audioPlayer.stop();
    }
    await audioPlayer.play(UrlSource(soundUrl));
    setState(() {
      isPlaying = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Lovince AI - The Founder - Lovince ™"),
      ),
      body: state == null
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Column(
                children: [
                  // Display cosmic spiral
                  Container(
                    margin: EdgeInsets.all(10),
                    child: CachedNetworkImage(
                      imageUrl: "data:image/png;base64,${state!['plot']}",
                      placeholder: (context, url) => CircularProgressIndicator(),
                      errorWidget: (context, url, error) => Icon(Icons.error),
                    ),
                  ),
                  // Play sound button
                  ElevatedButton(
                    onPressed: () => playSound("$serverUrl/sound/${state!['sound_file']}"),
                    child: Text(isPlaying ? "Stop Sound" : "Play Sound"),
                  ),
                  // Display SME insights
                  Padding(
                    padding: EdgeInsets.all(10),
                    child: Text(
                      state!['insights'],
                      style: TextStyle(fontSize: 16),
                    ),
                  ),
                  // Display metrics
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