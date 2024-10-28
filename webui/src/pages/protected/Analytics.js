import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Analytics from '../../features/analytics/index'

function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Analytics"}))
      }, [])


    return(
        <Analytics />
    )
}

export default InternalPage