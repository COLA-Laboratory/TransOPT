import React, { useState } from "react";
import { PlusOutlined } from '@ant-design/icons';
import { Button, Form, Select, Drawer, Modal } from "antd";
import DashboardStats from './DashboardStats'; // 确保路径正确
import UserGroupIcon from '@heroicons/react/24/outline/UserGroupIcon';
import UsersIcon from '@heroicons/react/24/outline/UsersIcon';
import CircleStackIcon from '@heroicons/react/24/outline/CircleStackIcon';
import CreditCardIcon from '@heroicons/react/24/outline/CreditCardIcon';

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function SelectAlgorithm({ SpaceRefiner, Sampler, Pretrain, Model, ACF, Normalizer }) {
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [form] = Form.useForm(); // Form instance to manage form submission in the drawer
  const [selectedValues, setSelectedValues] = useState({
    SpaceRefiner: SpaceRefiner[0]?.name || '',
    Sampler: Sampler[0]?.name || '',
    Pretrain: Pretrain[0]?.name || '',
    Model: Model[0]?.name || '',
    ACF: ACF[0]?.name || '',
    Normalizer: Normalizer[0]?.name || '',
    SpaceRefinerParameters: '',
    SpaceRefinerDataSelector: 'None',
    SpaceRefinerDataSelectorParameters: '',
    SamplerParameters: '',
    SamplerInitNum: '11',
    SamplerDataSelector: 'None',
    SamplerDataSelectorParameters: '',
    PretrainParameters: '',
    PretrainDataSelector: 'None',
    PretrainDataSelectorParameters: '',
    ModelParameters: '',
    ModelDataSelector: 'None',
    ModelDataSelectorParameters: '',
    ACFParameters: '',
    ACFDataSelector: 'None',
    ACFDataSelectorParameters: '',
    NormalizerParameters: '',
    NormalizerDataSelector: 'None',
    NormalizerDataSelectorParameters: '',
  });

  const [showDashboardStats, setShowDashboardStats] = useState(false);

  const handleDrawerSubmit = () => {
    form
      .validateFields()
      .then(formValues => {
        // Combine selectedValues with formValues
        const messageToSend = { ...selectedValues, ...formValues };
  
        fetch('http://localhost:5001/api/configuration/select_algorithm', {
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
            content: 'Submit successfully!',
          });
          setSelectedValues({ ...selectedValues, ...formValues }); // Update selectedValues with formValues
          setShowDashboardStats(true);
          form.resetFields(); // Reset the form fields after submission
          setDrawerVisible(false); // Close the drawer
        })
        .catch(error => {
          console.error('Error sending message:', error);
          Modal.error({
            title: 'Information',
            content: 'Error: ' + error.message,
          });
        });
      })
      .catch(info => {
        console.log('Validate Failed:', info);
      });
  };

  const statsData = [
    {title: "Space Refiner", value: selectedValues.SpaceRefiner || null, icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Space Refiner"},
    {title: "Sampler", value: selectedValues.Sampler || "N/A", icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Sampler"},
    {title: "Pretrain", value: selectedValues.Pretrain || null, icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Pretrain"},
    {title: "Model", value: selectedValues.Model || "N/A", icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Model"},
    {title: "Acquisition Function", value: selectedValues.ACF || "N/A", icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Acquisition Function"},
    {title: "Normalizer", value: selectedValues.Normalizer || "N/A", icon: <UserGroupIcon className='w-8 h-8'/>, description: "Details of Normalizer"},
  ].filter(stat => stat.value !== null && stat.value !== "N/A"); // Filter out entries with null or "N/A" values

  return (
    <>
      <Button type="primary" onClick={() => setDrawerVisible(true)} icon={<PlusOutlined />} style={{ width: "150px" }}>
        Start building
      </Button>

      <Drawer
        title="Select Algorithm"
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
          initialValues={selectedValues}
        >
          <Form.Item
            name="SpaceRefiner"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Space Refiner</span>}
            rules={[{ required: true, message: 'Please select a Space Refiner!' }]}
          >
            <Select
              showSearch
              placeholder="Space Refiner"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={SpaceRefiner.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="Sampler"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Sampler</span>}
            rules={[{ required: true, message: 'Please select a Sampler!' }]}
          >
            <Select
              showSearch
              placeholder="Sampler"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={Sampler.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="Pretrain"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Pretrain</span>}
            rules={[{ required: true, message: 'Please select a Pretrain!' }]}
          >
            <Select
              showSearch
              placeholder="Pretrain"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={Pretrain.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="Model"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Model</span>}
            rules={[{ required: true, message: 'Please select a Model!' }]}
          >
            <Select
              showSearch
              placeholder="Model"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={Model.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="ACF"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>ACF</span>}
            rules={[{ required: true, message: 'Please select an ACF!' }]}
          >
            <Select
              showSearch
              placeholder="ACF"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={ACF.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="Normalizer"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Normalizer</span>}
            rules={[{ required: true, message: 'Please select a Normalizer!' }]}
          >
            <Select
              showSearch
              placeholder="Normalizer"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={Normalizer.map(item => ({ value: item.name }))}
            />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" style={{ width: "150px" }}>
              Apply
            </Button>
          </Form.Item>
        </Form>
      </Drawer>

      {showDashboardStats && (
        <div className="grid lg:grid-cols-3 md:grid-cols-2 sm:grid-cols-1 gap-6 mt-6">
          {statsData.map((d, k) => (
            <DashboardStats key={k} {...d} colorIndex={k} />
          ))}
        </div>
      )}
    </>
  );
}

export default SelectAlgorithm;
