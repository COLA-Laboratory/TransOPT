import React from 'react';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ScatterController,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import TitleCard from '../../../components/Cards/TitleCard';

// 注册必要的组件
ChartJS.register(
  ScatterController,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);



const colors = [
  'rgba(255, 99, 132, 0.5)', // 红色
  'rgba(54, 162, 235, 0.5)', // 蓝色
  'rgba(255, 206, 86, 0.5)', // 黄色
];

// 转换数据为 Chart.js 可识别的格式



function Footprint({ ScatterData = {} }) {
  // 检查 ScatterData 是否为对象
  if (!ScatterData || typeof ScatterData !== 'object') {
    console.error('ScatterData is not a valid object:', ScatterData);
    return null; // 处理无效数据时返回 null 以避免进一步错误
  }

  const datasets = Object.keys(ScatterData).map((key, index) => ({
    label: key, // 数据集名称
    data: ScatterData[key].map(point => ({ x: point[0], y: point[1] })), // 将数据转换为 {x, y} 格式
    backgroundColor: colors[index % colors.length], // 设置不同的数据集颜色
    borderColor: colors[index % colors.length],
    pointRadius: 5, // 散点的大小
  }));

  // 定义图表选项
  const options = {
    scales: {
      x: {
        type: 'linear', // X轴类型为线性
      },
      y: {
        type: 'linear', // Y轴类型为线性
      },
    },
    plugins: {

      tooltip: {
        enabled: true, // 启用工具提示
      },
    },
  };

  // 图表数据
  const data = {
    datasets, // 使用转换后的数据集
  };

  return (
    <TitleCard title={"Footprint"}>
        <Scatter data={data} options={options}/>
    </TitleCard>
  );
}

export default Footprint;
