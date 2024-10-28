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
} from 'chart.js';

// 注册必要的组件
ChartJS.register(ScatterController, PointElement, LinearScale, Title, Tooltip, Legend);

function Footprint({ ScatterData = {} }) { // 提供默认值为空对象
  // 检查 ScatterData 是否为对象
  if (!ScatterData || typeof ScatterData !== 'object') {
    console.error('ScatterData is not a valid object:', ScatterData);
    return null; // 处理无效数据时返回 null 以避免进一步错误
  }

  // 转换数据为 Chart.js 可识别的格式
  const datasets = Object.keys(ScatterData).map((key, index) => ({
    label: key, // 数据集名称
    data: ScatterData[key].map(point => ({ x: point[0], y: point[1] })), // 将数据转换为 {x, y} 格式
    backgroundColor: 'rgba(75, 192, 192, 1)', // 设置散点的颜色，可以根据需要调整
    borderColor: 'rgba(75, 192, 192, 1)',
    pointRadius: 5, // 散点的大小
  }));

  // 定义图表选项
  const options = {
    plugins: {
      legend: {
        labels: {
          color: '#ffffff', // 设置图例文本颜色
        },
      },
      tooltip: {
        enabled: true, // 启用工具提示
      },
    },
    scales: {
      x: {
        type: 'linear', // X轴类型为线性
        position: 'bottom',
        ticks: {
          color: '#ffffff', // X轴标签颜色
        },
      },
      y: {
        type: 'linear', // Y轴类型为线性
        ticks: {
          color: '#ffffff', // Y轴标签颜色
        },
      },
    },
  };

  // 图表数据
  const data = {
    datasets, // 使用转换后的数据集
  };

  return <Scatter data={data} options={options} />;
}

export default Footprint;
