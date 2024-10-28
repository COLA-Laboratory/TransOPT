/** Icons are imported separatly to reduce build time */
import Squares2X2Icon from '@heroicons/react/24/outline/Squares2X2Icon'
import ChartBarIcon from '@heroicons/react/24/outline/ChartBarIcon'
import ChatBubbleLeftIcon from '@heroicons/react/24/outline/ChatBubbleLeftIcon'
import QuestionMarkCircleIcon from '@heroicons/react/24/outline/QuestionMarkCircleIcon'
import FolderOpenIcon from '@heroicons/react/24/outline/FolderOpenIcon'
import PlayIcon from '@heroicons/react/24/outline/PlayIcon'
import CogIcon from '@heroicons/react/24/outline/CogIcon'
import AdjustmentsHorizontalIcon from '@heroicons/react/24/outline/AdjustmentsHorizontalIcon'


const iconClasses = `h-6 w-6`
const submenuIconClasses = `h-5 w-5`

const routes = [

  {
    path: '/app/dashboard',
    icon: <Squares2X2Icon className={iconClasses}/>, 
    name: 'Dashboard',
  },

  {
    path: '/app/optimization', //no url needed as this has submenu
    icon: <CogIcon className={`${iconClasses} inline` }/>, // icon component
    name: 'Experiments', // name that appear in Sidebar
    submenu : [
      {
        path: '/app/optimization/problem',
        icon: <QuestionMarkCircleIcon className={submenuIconClasses}/>,
        name: 'Create New Experiment',
      },

      {
        path: '/app/optimization/selectdatasets',
        icon: <FolderOpenIcon className={submenuIconClasses}/>,
        name: 'Select Datasets',
      },
      {
        path: '/app/optimization/run',
        icon: <PlayIcon className={submenuIconClasses}/>,
        name: 'Run',
      },
    ]
  },

  {
    path: '/app/analytics', // url
    icon: <ChartBarIcon className={iconClasses}/>, // icon component
    name: 'Analytics', // name that appear in Sidebar
  },
  
  {
    path: '/app/chatopt',
    icon: <ChatBubbleLeftIcon className={iconClasses}/>, 
    name: 'ChatOpt',
  },

]

export default routes


