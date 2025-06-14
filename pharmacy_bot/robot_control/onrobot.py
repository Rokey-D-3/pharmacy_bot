#!/usr/bin/env python3

from pymodbus.client.sync import ModbusTcpClient as ModbusClient


class RG:
    def __init__(self, gripper, ip, port):
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E',
                                   baudrate=115200, timeout=1)
        if gripper not in ['rg2', 'rg6']:
            raise ValueError("Please specify either 'rg2' or 'rg6' for gripper.")
        self.gripper = gripper
        self.max_width = 1100 if gripper == 'rg2' else 1600
        self.max_force = 400 if gripper == 'rg2' else 1200
        self.open_connection()

    def open_connection(self):
        self.client.connect()

    def close_connection(self):
        self.client.close()

    def get_fingertip_offset(self):
        result = self.client.read_holding_registers(address=258, count=1, unit=65)
        return result.registers[0] / 10.0

    def get_width(self):
        result = self.client.read_holding_registers(address=267, count=1, unit=65)
        return result.registers[0] / 10.0

    def get_status(self):
        result = self.client.read_holding_registers(address=268, count=1, unit=65)
        status = format(result.registers[0], '016b')
        return [int(status[-(i + 1)]) for i in range(7)]

    def get_width_with_offset(self):
        result = self.client.read_holding_registers(address=275, count=1, unit=65)
        return result.registers[0] / 10.0

    def set_control_mode(self, command):
        self.client.write_register(address=2, value=command, unit=65)

    def set_target_force(self, force_val):
        self.client.write_register(address=0, value=force_val, unit=65)

    def set_target_width(self, width_val):
        self.client.write_register(address=1, value=width_val, unit=65)

    def close_gripper(self, force_val=400):
        params = [force_val, 0, 16]
        print("Start closing gripper.")
        self.client.write_registers(address=0, values=params, unit=65)

    def open_gripper(self, force_val=400):
        params = [force_val, self.max_width, 16]
        print("Start opening gripper.")
        self.client.write_registers(address=0, values=params, unit=65)

    def move_gripper(self, width_val, force_val=400):
        params = [force_val, width_val, 16]
        print("Start moving gripper.")
        self.client.write_registers(address=0, values=params, unit=65)
