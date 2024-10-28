import React from 'react';
import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/dist/index.css';
import './chatui-theme.css';

function ChatUI() {
  const { messages, appendMsg, setTyping } = useMessages([]);

  function handleSend(type, val) {
    if (type === 'text' && val.trim()) {
      appendMsg({
        type: 'text',
        content: { text: val },
        position: 'right',
        sender: 'you',
        time: new Date().toLocaleTimeString(),
      });

      const messageToSend = {
        type: 'text',
        content: { text: val },
      };

      fetch('http://localhost:5001/api/generate-yaml', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then((data) => {
          console.log('Message sent successfully:', data);
          appendMsg({
            type: 'text',
            content: { text: data.message },
            position: 'left',
            sender: 'robot',
            time: new Date().toLocaleTimeString(),
          });
        })
        .catch((error) => {
          console.error('Error sending message:', error);
          appendMsg({
            type: 'text',
            content: { text: 'There was an error processing your message.' },
            position: 'left',
            sender: 'robot',
            time: new Date().toLocaleTimeString(),
          });
        });
    }
  }

  function renderMessageContent(msg) {
    const { content, sender, position, time } = msg;
    const isLeft = position === 'left';

    // 头像路径
    const robotAvatar = '/robot.png'; // 替换为机器人头像的路径

    const bubbleStyle = {
      backgroundColor: isLeft ? '#E0E0E0' : '#81C784', // 左边灰色，右边绿色
      color: '#000',
      padding: '10px',
      borderRadius: '8px',
    };


    
    return (
      <div style={{ display: 'flex', alignItems: 'flex-start',justifyContent: isLeft ? 'flex-start' : 'flex-end', marginBottom: '10px' }}>
        {isLeft && (
          <img
            src={robotAvatar}
            alt="robot"
            style={{ width: '40px', height: '40px', borderRadius: '50%', marginRight: '10px' }}
          />
        )}
        <div>
          <div style={{ fontSize: '12px', color: '#555', marginBottom: '4px' }}>
            {sender === 'robot' ? 'ChatOPT' : 'You'}
          </div>
          <Bubble style={bubbleStyle}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span>{content.text}</span>
              <span style={{ fontSize: '12px', color: '#999', marginTop: '5px', alignSelf: 'flex-end' }}>
                {time}
              </span>
            </div>
          </Bubble>
        </div>
      </div>
    );
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
}

export default ChatUI;
