import React, { useState } from "react";
import { PlusOutlined } from '@ant-design/icons';
import { Button, Form, Input, Select, Modal, Drawer, Table } from "antd";

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function TaskTable({ tasks }) {
  return (
    <Table
      dataSource={tasks}
      pagination={false}
      rowKey="name"
      columns={[
        { title: '#', dataIndex: 'index', key: 'index' },
        { title: 'Task Name', dataIndex: 'name', key: 'name' },
        { title: 'Variables', dataIndex: 'num_vars', key: 'num_vars' },
        { title: 'Objectives', dataIndex: 'num_objs', key: 'num_objs' },
        { title: 'Fidelity', dataIndex: 'fidelity', key: 'fidelity' },
        { title: 'Workloads', dataIndex: 'workloads', key: 'workloads' },
        { title: 'Budget Type', dataIndex: 'budget_type', key: 'budget_type' },
        { title: 'Budget', dataIndex: 'budget', key: 'budget' }
      ]}
      locale={{
        emptyText: 'No task'
      }}
    />
    
  );
}

function SelectTask({ data, updateTable }) {
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [form] = Form.useForm(); // Form instance to manage form submission in the drawer
  const [tasks, setTasks] = useState([]); // State to store tasks added from Drawer

  const onFinish = (values) => {
    const messageToSend = tasks.map(task => ({
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
    // Send request to backend...
    fetch('http://localhost:5001/api/configuration/select_task', {
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
        });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        Modal.error({
          title: 'Information',
          content: 'Error: ' + error.message
        });
      });
  };

  const handleDrawerSubmit = () => {
    form
      .validateFields()
      .then(values => {
        console.log('Drawer form values:', values);

        // Add the task to the task list
        setTasks(prevTasks => [...prevTasks, values]);

        form.resetFields(); // Reset the form fields after submission
        setDrawerVisible(false); // Close the drawer
      })
      .catch(info => {
        console.log('Validate Failed:', info);
      });
  };

  return (
    <>
      <Form
        name="main_form"
        onFinish={onFinish}
        style={{ width: "100%" }}
        autoComplete="off"
      >
        <Form.List name="Tasks">
          {(fields, { add, remove }) => (
            <>
              <Form.Item
                name={['Experiment name']}
                style={{ marginBottom: '10px' }} // Add margin bottom
              >
                <Input
                  placeholder="Experiment name"
                  style={{
                    width: '300px', // Full width of the container
                    fontSize: '32px', // Font size
                    resize: 'vertical', // Allow vertical resizing only
                  }}
                />
              </Form.Item>

              <Form.Item
                name={['experiment_description']}
                style={{ marginBottom: '16px' }} // Add margin bottom
              >
                <Input.TextArea
                  placeholder="Type the description of the experiment"
                  style={{
                    width: '100%', // Full width of the container
                    height: '200px', // Height of the text area
                    fontSize: '16px', // Font size
                    resize: 'vertical', // Allow vertical resizing only
                  }}
                />
              </Form.Item>
            </>
          )}
        </Form.List>
        <Form.Item>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button type="primary" htmlType="submit" style={{
              width: "150px",
              backgroundColor: 'rgb(53, 162, 235)',
            }}>
              Submit
            </Button>

            <Button onClick={() => setDrawerVisible(true)} icon={<PlusOutlined />} style={{
              width: "150px",
              borderColor: 'black',
            }}>
              Add new task
            </Button>
          </div>
        </Form.Item>
      </Form>

      <Drawer
        title="Add new task"
        placement="right"
        closable={true}
        onClose={() => setDrawerVisible(false)}
        visible={drawerVisible}
        width={720}
      >
        <Form
          form={form}
          name="drawer_form"
          onFinish={handleDrawerSubmit}
          style={{ width: "100%" }}
          autoComplete="off"
        >
          <Form.Item
            name="name"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Problem Name</span>}
            rules={[{ required: true, message: 'Please select a problem name!' }]}
          >
            <Select
              showSearch
              placeholder="problem name"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={data.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="num_vars"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Number of Variables</span>}
            rules={[{ required: true, message: 'Please enter the number of variables!' }]}
          >
            <Input placeholder="number of variables" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="num_objs"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Number of Objectives</span>}
            rules={[{ required: true, message: 'Please select the number of objectives!' }]}
          >
            <Input placeholder="number of objectives" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="fidelity"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Fidelity</span>}
            rules={[{ required: false, message: 'Please select fidelity!' }]}
          >
            <Select
              placeholder="fidelity"
              options={[]}
              style={{ fontSize: '14px', width: '300px' }}
            />
          </Form.Item>
          <Form.Item
            name="workloads"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Workloads</span>}
            rules={[{ required: true, message: 'Please specify workloads!' }]}
          >
            <Input placeholder="specify workloads" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="budget_type"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Budget Type</span>}
            rules={[{ required: true, message: 'Please select budget type!' }]}
          >
            <Select
              placeholder="budget type"
              style={{ fontSize: '14px', width: '200px' }}
              options={[
                { value: "function evaluations" },
                { value: "hours" },
                { value: "minutes" },
                { value: "seconds" },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="budget"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Budget</span>}

            rules={[{ required: true, message: 'Please enter the budget!' }]}
          >
            <Input placeholder="budget" style={{ fontSize: '14px', width: '200px' }} />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" style={{ width: "150px", backgroundColor: 'rgb(53, 162, 235)' }}>
              Add
            </Button>
          </Form.Item>
        </Form>
      </Drawer>

      <Form>
        <Form.Item>
          <TaskTable tasks={tasks} />
        </Form.Item>
      </Form>
    </>
  );
}

export default SelectTask;
