import json
import logging

from typing import Optional, List
from pathlib import Path

from zaber_motion import Units
from zaber_motion.ascii import Connection
from zaber_motion.exceptions.connection_failed_exception import ConnectionFailedException
from zaber_motion.exceptions.movement_failed_exception import MovementFailedException
from zaber_motion.exceptions.bad_data_exception import BadDataException
# TODO standardize comments

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject


# TODO: Auto-populate well positions and well plate types

class ZaberCon(QObject):
    """Communicate with Zaber devices over serial to move the stages
        Note that this class is using the zaber_motion.binary library instead of 
        zaber_motion.ascii because of older T-series devices that do not support the ASCII Protocol
    """
    error = pyqtSignal(str)
    manual = pyqtSignal(bool)

    update_h = pyqtSignal(float)
    update_x = pyqtSignal(float)
    update_y = pyqtSignal(float)    

    update = {
        'h': update_h,
        'x': update_x,
        'y': update_y,
    }

    def __init__(self, env='prod'):
        """Setup the serial connection with the zaber device

        :param config: Path to the config folder containing
                    zaber_config.json file defining
                    Zaber specific parameters
        :param env: The environment to run the Zaber Controller.
        :type env: string, either 'prod' or 'dev'
        """

        super().__init__()
        
        # Load .json config
        with open(self.get_config_path(), 'r') as f:
            zc_cfg = json.load(f)
        self.config = zc_cfg['zaber_config']

        self.zaber = None
        self.stage_alias = {}
        self.env = env
        self._connect()

    def get_config_path(self):
        file = Path(__file__).parent / 'configs' / 'local' / 'zaber.json'
        if file.exists():
            return file
        
        raise(FileNotFoundError(
            f"Missing local (untracked in Git) zaber.json config file in {file.parent}. "
            "Copy default version (tracked in Git) from configs\\default\\"
        ))

    def _connect(self):
        """Create a serial communication with the zaber devices

        :raises ConnectionFailedException: Logs critical if the connection fails
        """
        
        try:
            if self.env == 'prod':
                logging.info('Establishing connection with Zaber devices')
                self.zaber = Connection.open_serial_port(self.config['port'])
                logging.info('Zaber devices successfully connected')

                self.controller = self.zaber.detect_devices()[0]
                # Set the names and velocities for each axis
                self._set_axis()

                self.home_all()
            elif self.env == 'dev':
                logging.info('Establishing connection with mock Zaber devices')
                self.zaber = Zaber(self.config['port'])
                logging.info('Zaber devices successfully connected')

                self.controller = self.zaber.detect_devices()[0]
                # Set the names for each axis
                self._set_axis(self.zaber.detect_devices())

                self.home_all()
        except ConnectionFailedException:
            logging.critical("Could not make connection to zaber stage")
            raise

    def _set_axis(self):
        """Set the x, y, p stage dictionary variables based off the peripheral name

        :param stage: zaber x, y, p stage
        :type stage: tuple of zaber device objects
        """

        for alias in ['h']:
            try:
                stage = self.controller.get_axis(int(self.config['axis'][alias]))
            except ValueError:
                # Ignore _description entry in .json
                break
            name = stage.identity.peripheral_name

            self.stage_alias[alias] = stage
            self.stage_alias[alias].settings.set("maxspeed", self.config['max_speed'][alias], Units.VELOCITY_MICROMETRES_PER_SECOND)

            logging.info(f'Set stage {name} as {alias} axis')

        logging.info('Done setting all axes')

    def home_all(self):
        """Home either all or a subset of the devices

        The devices include the x, y, p stages. The order in which
        it homes is dependent on the list passed. The order is important 
        to ensure the device does not crash while homing.

        :param arm: list of the devices to home in the desired sequence,
                    defaults to None, if None homes everything
        :type arm: list of str, optional
        :raises: Any Zaber exception requires restart and reinitialization of Zaber connection
        """

        wait = True
        for axis in ['h']:
            try:
                self.move_arm(axis, wait=wait)
                wait = False # only wait for h axis
            except:
                raise
    
    def move_arm(
        self,
        arm: str,
        dist: Optional[float]=None,
        is_relative: bool=False,
        speed: Optional[int]=None,
        wait: bool=True,
    ):
        """Move any arm 'x','y','h' by a fixed amount

        :param arm: The arm to move x' or 'y' or 'h'
        :type arm: str
        :param dist: The distance to move in um, if None: home arm, defaults to None
        :type dist: float, optional
        :param is_relative: True: move a relative distance, False: move an absolute distance,
                    defaults to False
        :type is_relative: bool, optional
        :raises MovementFailedException: Logs if the desired position is not reached
        :raises ConnectionFailedException: Logs if the zaber connection fails
        """

        device_arm = self.stage_alias[arm]
        name = device_arm.identity.peripheral_name

        if speed is None:
            speed = self.config['max_speed'][arm]
        if speed > self.config['max_speed'][arm]:
            logging.warning(f"Capping '{arm}' speed at {self.config['max_speed'][arm]:.2f} um/s (requested speed = {speed:.2f} um/s) ")
            speed = self.config['max_speed'][arm]

        try:
            if dist is None:
                logging.debug(f"Homing '{arm}' arm ({name})")
                pos = self.config['home'][arm]
                device_arm.move_absolute(
                    pos,
                    Units.LENGTH_MICROMETRES,
                    velocity=speed,
                    velocity_unit=Units.VELOCITY_MICROMETRES_PER_SECOND,
                    wait_until_idle=wait,
                )
                self.get_signal(arm).emit(pos)
            elif is_relative:
                logging.debug(f"Moving '{arm}' arm ({name}) by {dist} um (rel)")
                pos = self.get_pos(arm) + dist
                device_arm.move_relative(
                    dist,
                    Units.LENGTH_MICROMETRES,
                    velocity=speed,
                    velocity_unit=Units.VELOCITY_MICROMETRES_PER_SECOND,
                    wait_until_idle=wait,
                )
                self.get_signal(arm).emit(pos)
            else:
                logging.debug(f"Moving '{arm}' arm ({name}) to {dist} um (abs)")
                pos = dist
                device_arm.move_absolute(
                    dist,
                    Units.LENGTH_MICROMETRES,
                    velocity=speed,
                    velocity_unit=Units.VELOCITY_MICROMETRES_PER_SECOND,
                    wait_until_idle=wait,
                )
                self.get_signal(arm).emit(pos)

        except BadDataException:
            msg = f"Invalid '{arm}' arm position or speed requested ({pos:.2f} um at {speed:.2f} um/s)"
            logging.warning(msg)
            self.error.emit(msg)
        except MovementFailedException:
            msg = f"Failed to complete move '{arm}' arm ({pos:.2f} um at {speed:.2f} um/s)"
            logging.error(msg)
            self.error.emit(msg)
        except ConnectionFailedException:
            msg = 'Zaber connection failed'
            logging.error(msg)
            self.error.emit(msg)

    def manual_drive(self, enabled: bool):
        for ax in self.stage_alias.values():
            ax.settings.set('knob.enable', enabled) 
        logging.info(("Enabled" if enabled else "Disabled") + " manual drive on all axes")

        self.manual.emit(enabled)

    def get_signal(self, arm: str) -> pyqtSignal:
        return getattr(self, f'update_{arm}')

    def get_pos(self, arm: str) -> float:
        """returns the position of the zaber stage

        :param arm: The arm to move x' or 'y' or 'h'
        :type arm: str
        :return: The stage location position in um
        :rtype: float
        """
        
        device_arm = self.stage_alias[arm]
        name = device_arm.identity.peripheral_name

        try:
            curr_pos = device_arm.get_position(unit=Units.LENGTH_MICROMETRES)
            self.get_signal(arm).emit(curr_pos)
            return curr_pos
        except ConnectionFailedException:
            logging.critical('Zaber connection failed')

    def close(self):
        """Closes the serial Connection
        """

        # Re-enable manual control
        self.manual_drive(True)

        self.zaber.close()
        logging.info('Closed Zaber device connection')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    zc = ZaberCon()
    # zc.move_arm('h', 2, speed=100000.0, is_relative=True)

    print('Manual control disabled. Run script to completion to re-enable manual control.')
    zc.manual_drive(False)

    # input("Click enter to move Z axis via software control")
    # zc.move_arm('h', 100, speed=100000.0, is_relative=False)

    input("Click enter to re-enable knobs and disconnnect drom zaber")
    zc.close()
