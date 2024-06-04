import React from 'react';
import { Row, Col } from 'reactstrap';
import s from '../optimizer/Optimizer.module.scss';
import Widget from '../../components/Widget/Widget';
import ChatUI from './ChatUI'

class Chatbot extends React.Component {

    render() {
        return (
            <div className={s.root}>
              <h1 className="page-title">
                Chat - <span className="fw-semi-bold">bot</span>
              </h1>
                <Row>
                <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      Chat<span className="fw-semi-bold">TOS</span>
                    </h5>
                  }
                >
                  <div className={s.chatui}>
                    <ChatUI />
                  </div>
                </Widget>
                </Col>
                <Col lg={12} sm={12}>
                </Col>
              </Row>
            </div>
        );
    }
}

export default Chatbot;