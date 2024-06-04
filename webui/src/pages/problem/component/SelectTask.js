import React, { useState } from "react";

import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import {
    Button,
    Form,
    Input,
    Space, 
    Select,
    Modal,
} from "antd";

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function SelectDim({anyDim, num_vars, name, restField}) {
  if (anyDim) {
      return (
        <Form.Item
           {...restField}
           name={[name, 'num_vars']}
         >
        <Input placeholder="number of variables" />
        </Form.Item>
      )
  } else {
      return (
        <Form.Item
           {...restField}
           name={[name, 'num_vars']}
         >
           <Select
             placeholder="number of variables"
             options={num_vars}
           />
         </Form.Item>
      )
  }
}

function ATask({key, name, restField, remove, data}) {
    const [selectedTask, setSelectedTask] = useState(data[0])
    
    function handleNameChange(value) {
        console.log(`selected ${value}`);
        const task = data.find(task => task.name === value)
        setSelectedTask(task)
    }

    return (
        <Space className="space" key={key} style={{ display: 'flex', marginBottom: 1 }} align="baseline">
           <Form.Item
             {...restField}
             name={[name, 'name']}
           >
             <Select
               showSearch
               placeholder="problem name"
               optionFilterProp="value"
               filterOption={filterOption}
               options={data.map(item => ({ value: item.name}))}
               onChange={handleNameChange}
               style={{minWidth:"100px"}}
             />
           </Form.Item>

           <SelectDim anyDim={selectedTask.anyDim} num_vars={selectedTask.num_vars.map(item => ({value: item}))} name={name} restField={restField} />
           <Form.Item
             {...restField}
             name={[name, 'num_objs']}           
            >
            <Select
              placeholder="number of objectives"
              options={selectedTask.num_objs.map(item => ({value: item}))}
            />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'fidelity']}
           >
             <Select
               placeholder="fidelity"
               options={selectedTask.fidelity.map(item => ({value: item}))}
             />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'workloads']}
           >
             <Input placeholder="specify workloads" />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'budget_type']}
           >
             <Select
               placeholder="budget type"
               options={[ {value: "evaluations"},
                          {value: "hours"},
                          {value: "minutes"},
                          {value: "seconds"},
                        ]}
             />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'budget']}
           >
             <Input placeholder="budget" />
           </Form.Item>
           <MinusCircleOutlined style={{color: 'white'}} onClick={() => remove(name)} />
        </Space>
    )
}

function SelectTask({data, updateTable}) {
  const onFinish = (values) => {
    // 构造要发送到后端的数据
    const messageToSend = values.Tasks.map(task => ({
      name: task.name,
      num_vars: parseInt(task.num_vars),
      num_objs: task.num_objs,
      fidelity: task.fidelity,
      workloads: task.workloads,
      budget_type: task.budget_type,
      budget: task.budget,
    }));
    updateTable(messageToSend);
    console.log('Request data:', messageToSend);
    // 向后端发送请求...
    fetch('http://localhost:5000/api/configuration/select_task', {
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
      .then(succeed => {
        console.log('Message from back-end:', succeed);
        Modal.success({
          title: 'Information',
          content: 'Submit successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        var errorMessage = error.error;
        Modal.error({
          title: 'Information',
          content: 'error:' + errorMessage
        })
      });
  };

  return (
    <Form
      name="dynamic_form_nest_item"
      onFinish={onFinish}
      style={{ width:"100%" }}
      autoComplete="off"
    >
      <Form.List name="Tasks">
        {(fields, { add, remove }) => (
          <>
            <div>
            {fields.map(({ key, name, ...restField }) => (
              <ATask key={key} name={name} restField={restField} remove={remove} data={data} />
            ))}
            </div>
            <Form.Item style={{marginTop:10}}>
              <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{width:"150px"}}>
                Add Problem
              </Button>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" style={{width:"120px"}}>
                Submit
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </Form>
  )
}

export default SelectTask;