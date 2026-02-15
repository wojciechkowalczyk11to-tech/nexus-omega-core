import React, { useState, useCallback, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, SafeAreaView, Platform, StatusBar } from 'react-native';
import { GiftedChat, Bubble, InputToolbar, Send } from 'react-native-gifted-chat';
import { Settings, Menu, Zap, ArrowUp, Plus } from 'lucide-react-native';

// Nexus Omega Mobile App
// Cross-platform (iOS/Android) interface for the AI Agent

const THEME = {
  background: '#0d1117', // gray-950
  surface: '#161b22',    // gray-900
  primary: '#2563eb',    // blue-600
  text: '#e6edf3',       // gray-100
  secondaryText: '#8b949e',
  border: '#30363d'
};

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    setMessages([
      {
        _id: 1,
        text: 'Hello! I am Nexus Omega. How can I help you today?',
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'Nexus Omega',
          avatar: 'https://ui-avatars.com/api/?name=Nexus+Omega&background=2563eb&color=fff',
        },
      },
    ]);
  }, []);

  const onSend = useCallback((messages = []) => {
    setMessages(previousMessages => GiftedChat.append(previousMessages, messages));
    setIsTyping(true);

    // Simulate Backend Response
    // In a real build, this would connect to the backend API
    const userMessage = messages[0].text;
    
    setTimeout(() => {
      const response = {
        _id: Math.round(Math.random() * 1000000),
        text: `I received your request: "${userMessage}".\n\nI am processing this using the Nexus Omega core engine.`,
        createdAt: new Date(),
        user: {
          _id: 2,
          name: 'Nexus Omega',
          avatar: 'https://ui-avatars.com/api/?name=Nexus+Omega&background=2563eb&color=fff',
        },
      };
      
      setMessages(previousMessages => GiftedChat.append(previousMessages, [response]));
      setIsTyping(false);
    }, 1500);
  }, []);

  const renderBubble = (props) => {
    return (
      <Bubble
        {...props}
        wrapperStyle={{
          right: {
            backgroundColor: THEME.primary,
            borderRadius: 12,
            borderBottomRightRadius: 2,
          },
          left: {
            backgroundColor: THEME.surface,
            borderRadius: 12,
            borderBottomLeftRadius: 2,
          }
        }}
        textStyle={{
          right: { color: '#fff' },
          left: { color: THEME.text },
        }}
      />
    );
  };

  const renderSend = (props) => {
    return (
      <Send {...props}>
        <View style={styles.sendButton}>
          <ArrowUp color="#fff" size={20} />
        </View>
      </Send>
    );
  };

  const renderInputToolbar = (props) => {
    return (
      <InputToolbar
        {...props}
        containerStyle={styles.inputToolbar}
        primaryStyle={{ alignItems: 'center' }}
      />
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={THEME.background} />
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.iconButton}>
          <Menu color={THEME.secondaryText} size={24} />
        </TouchableOpacity>
        
        <View style={styles.headerTitleContainer}>
          <Zap color={THEME.primary} size={20} fill={THEME.primary} />
          <Text style={styles.headerTitle}>Nexus Omega</Text>
        </View>

        <TouchableOpacity style={styles.iconButton}>
          <Plus color={THEME.secondaryText} size={24} />
        </TouchableOpacity>
      </View>

      {/* Chat Interface */}
      <GiftedChat
        messages={messages}
        onSend={messages => onSend(messages)}
        user={{ _id: 1 }}
        renderBubble={renderBubble}
        renderSend={renderSend}
        renderInputToolbar={renderInputToolbar}
        isTyping={isTyping}
        alwaysShowSend
        scrollToBottom
        listViewProps={{
          style: { backgroundColor: THEME.background },
          contentContainerStyle: { flexGrow: 1, paddingBottom: 10 }
        }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.background,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: THEME.background,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  headerTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    color: THEME.text,
    fontSize: 18,
    fontWeight: 'bold',
  },
  iconButton: {
    padding: 8,
  },
  inputToolbar: {
    backgroundColor: THEME.surface,
    borderTopColor: THEME.border,
    borderTopWidth: 1,
    paddingVertical: 4,
    marginHorizontal: 10,
    marginBottom: 6,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  sendButton: {
    marginBottom: 4,
    marginRight: 4,
    backgroundColor: THEME.primary,
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
