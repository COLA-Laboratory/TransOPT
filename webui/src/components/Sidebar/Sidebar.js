import React from 'react';
import cx from 'classnames';
import PropTypes from 'prop-types';
import {connect} from 'react-redux';
import {withRouter} from 'react-router-dom';
import {dismissAlert} from '../../actions/alerts';
import s from './Sidebar.module.scss';
import LinksGroup from './LinksGroup';

import {changeActiveSidebarItem} from '../../actions/navigation';
import {logoutUser} from '../../actions/user';
import HomeIcon from '../Icons/SidebarIcons/HomeIcon';
import TypographyIcon from '../Icons/SidebarIcons/TypographyIcon';
import TablesIcon from '../Icons/SidebarIcons/TablesIcon';
import NotificationsIcon from '../Icons/SidebarIcons/NotificationsIcon';


class Sidebar extends React.Component {
    static propTypes = {
        sidebarStatic: PropTypes.bool,
        sidebarOpened: PropTypes.bool,
        dispatch: PropTypes.func.isRequired,
        activeItem: PropTypes.string,
        location: PropTypes.shape({
            pathname: PropTypes.string,
        }).isRequired,
    };

    static defaultProps = {
        sidebarStatic: false,
        activeItem: '',
    };

    constructor(props) {
        super(props);

        this.doLogout = this.doLogout.bind(this);
    }

    componentDidMount() {
        this.element.addEventListener('transitionend', () => {
            if (this.props.sidebarOpened) {
                this.element.classList.add(s.sidebarOpen);
            }
        }, false);
    }

    componentWillReceiveProps(nextProps) {
        if (nextProps.sidebarOpened !== this.props.sidebarOpened) {
            if (nextProps.sidebarOpened) {
                this.element.style.height = `${this.element.scrollHeight}px`;
            } else {
                this.element.classList.remove(s.sidebarOpen);
                setTimeout(() => {
                    this.element.style.height = '';
                }, 0);
            }
        }
    }

    dismissAlert(id) {
        this.props.dispatch(dismissAlert(id));
    }

    doLogout() {
        this.props.dispatch(logoutUser());
    }

    render() {
        return (
            <nav
                className={cx(s.root)}
                ref={(nav) => {
                    this.element = nav;
                }}
            >
                <header className={s.logo}>
                    <span className="fw-bold">Transfer Optimization System</span>
                </header>
                <ul className={s.nav}>
                    <LinksGroup
                        onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                        activeItem={this.props.activeItem}
                        header="Dashboard"
                        isHeader
                        iconName={<HomeIcon className={s.menuIcon} />}
                        link="/app/Dashboard"
                        index="dashboard"
                    />
                    <h5 className={[s.navTitle, s.groupTitle].join(' ')}>FUNCTIONS</h5>
                    <LinksGroup
                        onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                        activeItem={this.props.activeItem}
                        header="Specify Problem"
                        isHeader
                        iconName={<TypographyIcon className={s.menuIcon} />}
                        link="/app/Problem"
                        index="problem"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                        activeItem={this.props.activeItem}
                        header="Composite Algorithm"
                        isHeader
                        iconName={<TypographyIcon className={s.menuIcon} />}
                        link="/app/Optimizer"
                        index="optimizer"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                        activeItem={this.props.activeItem}
                        header="Select Data"
                        isHeader
                        iconName={<TypographyIcon className={s.menuIcon} />}
                        link="/app/Dataselector"
                        index="dataselector"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                        activeItem={this.props.activeItem}
                        header="Run Experiment"
                        isHeader
                        iconName={<TypographyIcon className={s.menuIcon} />}
                        link="/app/Run"
                        index="run"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                        activeItem={this.props.activeItem}
                        header="Compare Results"
                        isHeader
                        iconName={<TablesIcon className={s.menuIcon} />}
                        link="/app/Comparision"
                        index="comparision"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                        activeItem={this.props.activeItem}
                        header="Chatbot"
                        isHeader
                        iconName={<NotificationsIcon className={s.menuIcon}/>}
                        link="/app/Chatbot"
                        index="ui"
                    />
                    
                </ul>
            </nav>
        );
    }
}

function mapStateToProps(store) {
    return {
        sidebarOpened: store.navigation.sidebarOpened,
        sidebarStatic: store.navigation.sidebarStatic,
        alertsList: store.alerts.alertsList,
        activeItem: store.navigation.activeItem,
    };
}

export default withRouter(connect(mapStateToProps)(Sidebar));
