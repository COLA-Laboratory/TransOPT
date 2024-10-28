import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import TitleCard from '../../../components/Cards/TitleCard';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function BarChart({ ImportanceData }){

    const options = {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          }
        },
      };
      
      const labels = ['x1', 'x2', 'x3', 'x4'];
      
      const data = {
        labels,
        datasets: [
          {
            label: 'Importance level',
            data: labels.map(() => { return Math.random() * 0.1 + 0.7 }),
            backgroundColor: 'rgba(255, 99, 132, 1)',
          },
        ],
      };

    return(
      <TitleCard title={"Importance of variables"}>
            <Bar options={options} data={data} />
      </TitleCard>

    )
}


export default BarChart