import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Experiment from '../../features/experiment/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Create New Experiment" }))
      }, [])


    return(
        <Experiment />
    )
}

export default InternalPage