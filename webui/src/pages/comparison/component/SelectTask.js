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


function ASearch({key, name, restField, remove, selections}) {

    return (
        <Space key={key} className="space" style={{ marginBottom: 1 }} align="baseline">
           <Form.Item
             {...restField}
             name={[name, 'TaskName']}
           >
             <Input placeholder="TaskName" style={{ minWidth: 93 }} />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'NumObjs']}
           >
             <Input placeholder="NumObjs" style={{ minWidth: 83 }} />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'NumVars']}
           >
             <Input placeholder="NumVars" style={{ minWidth: 83 }} />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Fidelity']}
           >
             <Input placeholder="Fidelity" style={{ minWidth: 70 }} />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Workload']}
           >
             <Input placeholder="Workload" style={{ minWidth: 85 }} />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Seed']}
           >
             <Input placeholder="Seed" style={{ minWidth: 60 }} />
           </Form.Item>
           
           <Form.Item
             {...restField}
             name={[name, 'Refiner']}
           >
             <Select
               placeholder="Refiner"
               options={selections.Refiner.map(item => ({value: item}))}
             />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Sampler']}
           >
             <Select
               placeholder="Sampler"
               options={selections.Sampler.map(item => ({value: item}))}
             />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Pretrain']}
           >
             <Select
               placeholder="Pretrain"
               options={selections.Pretrain.map(item => ({value: item}))}
             />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Model']}
           >
             <Select
               placeholder="Model"
               options={selections.Model.map(item => ({value: item}))}
             />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'ACF']}
           >
             <Select
               placeholder="ACF"
               options={selections.ACF.map(item => ({value: item}))}
             />
           </Form.Item>

           <Form.Item
             {...restField}
             name={[name, 'Normalizer']}
           >
             <Select
               placeholder="Normalizer"
               options={selections.Normalizer.map(item => ({value: item}))}
             />
           </Form.Item>

           <MinusCircleOutlined style={{color: 'white'}} onClick={() => remove(name)} />
        </Space>
    )
}

function SelectTask({selections, handleClick}) {
  console.log("SelectTask recieve info:" , selections);
  return (
    <Form
      name="dynamic_form_nest_item"
      onFinish={handleClick}
      style={{ width:"100%" }}
      autoComplete="off"
    >
      <Form.List name="Tasks">
        {(fields, { add, remove }) => (
          <>
            <div style={{ overflowY: 'auto', maxHeight: '200px' }}>
            {fields.map(({ key, name, ...restField }) => (
              <ASearch key={key} name={name} restField={restField} remove={remove} selections={selections} />
            ))}
            </div>
            <Form.Item style={{marginTop:20}}>
              <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{width:"120px"}}>
                Add
              </Button>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" style={{width:"120px"}}>
                Search
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </Form>
  )
}

export default SelectTask;