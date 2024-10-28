import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Algorithm from '../../features/algorithm/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Choose Algorithm Objects"}))
      }, [])


    return(
        <Algorithm />
    )
}

export default InternalPage