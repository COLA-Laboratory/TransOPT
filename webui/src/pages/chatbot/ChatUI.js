import React from 'react';

import Chat, { Bubble, useMessages, Icon } from '@chatui/core';
import '@chatui/core/dist/index.css';
import './chatui-theme.css'

function ChatUI () {
  const { messages, appendMsg, setTyping } = useMessages([]);

  function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
      });

      const messageToSend = {
        type: 'text',
        content: { text: val },
      };

      fetch('http://localhost:5000/api/generate-yaml', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        console.log('Message sent successfully:', data);
        appendMsg({
          type: 'text',
          content: { text: data.message }, 
          position: 'left',
        });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        appendMsg({
          type: 'text',
          content: { text: 'There was an error processing your message.' },
          position: 'left',
        });
      });
    }
  }

  function renderMessageContent(msg) {
    const { content } = msg;
    return <Bubble content={content.text} />;
  }

  return (
    <Chat
      messages={messages}
      renderMessageContent={renderMessageContent}
      onSend={handleSend}
      placeholder='Send a message...'
      locale='en-US'
    />
  );
};

export default ChatUI