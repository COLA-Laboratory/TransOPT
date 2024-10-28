
import React from 'react';
import { Row, Col } from 'reactstrap';
import TitleCard from "../../components/Cards/TitleCard"
import ChatUI from './components/ChatUI'

class ChatBot extends React.Component {

    render() {
        return (
          <div>
          <div className="mt-4 w-[1400px] p-4 bg-gray-100">

                <TitleCard
                  title={
                    <h5>
                      ChatOpt
                    </h5>
                  }
                >
                    <ChatUI />
                </TitleCard>
            </div>
          </div>
        );
    }
}


export default ChatBot