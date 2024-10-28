import re

import docker

from transopt.utils.log import logger
from transopt.utils.path import get_absolut_path


class AbsolutContainerManager:
    def __init__(self):
        self.docker_image = "absolut_image"
        self.container_name = "absolut_container"
        self.antigen_dir = get_absolut_path()

        self.docker_client = docker.from_env()
    
    def prepare_antigen(self, antigen):
        logger.info(f"Preparing antigen data for {antigen}...")

        try:
            container = self.docker_client.containers.run(
                self.docker_image,
                f"prepare_antigen.sh {antigen}",
                volumes={self.antigen_dir: {'bind': '/usr/local/Absolut', 'mode': 'rw'}},
                remove=True,
                detach=True,
            )
            container.wait()
            logger.debug(f"Successfully downloaded data for antigen: {antigen}")
        except docker.errors.DockerException as e:
            logger.error(f"Error running container for antigen {antigen}: {e}")
            raise e

    def predict_energy(self, antigen, sequence):
        """
        预测给定 antigen 和序列的能量
        :param antigen: 要预测的 antigen
        :param sequence: 要预测的序列
        :return: 预测的能量结果
        """
        if not self.validate_sequence(sequence):
            raise ValueError("Invalid sequence. Please ensure it is 11 characters long and only contains valid amino acids.")
        
        self.prepare_antigen(antigen)

        logger.info(f"Predicting energy for antigen {antigen} with sequence {sequence}...")

        try:
            container = self.docker_client.containers.run(
                self.docker_image,
                f"AbsolutNoLib singleBinding {antigen} {sequence}",
                volumes={self.antigen_dir: {'bind': '/usr/local/Absolut', 'mode': 'rw'}},
                remove=True,
                detach=True,
            )

            output = container.logs(stream=True)
            energy_result = None
            pattern = re.compile(rf"{sequence}\s+{sequence}\s+(-?\d+(\.\d+)?)\s")
            
            for line in output:
                decoded_line = line.decode().strip()
                logger.debug(decoded_line)
                match = pattern.search(decoded_line)
                if match:
                    energy_result = float(match.group(1))
                    break

            result = container.wait()
            exit_code = result['StatusCode']
            if exit_code == 0:
                logger.debug(f"Successfully predicted energy for antigen: {antigen}")
                return energy_result
            else:
                logger.error(f"Container exited with error code {exit_code} for antigen: {antigen}")
                raise RuntimeError(f"Container exited with error code {exit_code}")

        except docker.errors.DockerException as e:
            logger.error(f"Error running container for antigen {antigen} with sequence {sequence}: {e}")
            raise e

    def validate_sequence(self, sequence):
        """
        验证序列的长度和字符是否合法
        """
        valid_chars = set(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
        
        if len(sequence) != 11:
            logger.error(f"Sequence length is {len(sequence)}, but it must be 11.")
            return False
        
        if not all(char in valid_chars for char in sequence):
            logger.error(f"Sequence contains invalid characters. Valid characters are: {valid_chars}.")
            return False
        
        return True
    
    def get_available_workloads(self):
        workloads =  [
            'S3A3_A', 'S3B3_A', 'S3C3_A', 'S3D3_A', 'S3E3_A', 'S3F3_A', 'S3G3_A', 'S3H3_A',
            'S2A2_A', 'S2B2_A', 'S2C2_A', 'S2D2_A', 'S2E2_A', 'S2F2_A', 'S2G2_A', 'S2H2_A',
            'S4A4_A', 'S4B4_A', 'S4C4_A', 'S4D4_A', 'S4E4_A', 'S4F4_A', 'S4G4_A', 'S4H4_A',
            'S5A5_A', 'S5B5_A', 'S5C5_A', 'S5D5_A', 'S5E5_A', 'S5F5_A', 'S5G5_A', 'S5H5_A',
            '1FNS_A', '1ADQ_A', '1FSK_A', '1FBI_X', '1H0D_C', '1NSN_S', '1OB1_C', '1OSP_O',
            '1PKQ_J', '1TQB_A', '1WEJ_F', '1YJD_C', '1ZTX_E', '1RJL_C', '2B2X_A', '2DD8_S',
            '2HFG_R', '2IH3_C', '2JEL_P', '2R29_A', '2R56_A', '2UZI_R', '2VXQ_A', '2VXT_I',
            '2W9E_A', '2XWT_C', '2YC1_C', '5DO2_B', '3B9K_A', '3BGF_S', '3EFD_K', '3G04_C',
            '3GI9_C', '3HI6_A', '3JBQ_B', '3KJ4_A', '3KR3_D', '3KS0_J', '3L5W_I', '3MJ9_A',
            '3NH7_A', '3Q3G_E', '3R08_E', '3RAJ_A', '3RKD_A', '3RVV_A', '3SKJ_E', '3VRL_C',
            '3WD5_A', '4AEI_A', '4H88_A', '4Hj0_A', '4HJ0_B', '4IJ3_A', '4K24_A', '4K3J_A',
            '4KI5_M', '4KXZ_A', '4KXZ_B', '4KXZ_E', '4N9G_C', '4NP4_A', '4OKV_E', '4PP1_A',
            '4QCI_D', '4QNP_A', '4R9Y_D', '4WV1_F', '4YPG_C', '4ZFG_A', '4ZFO_F', '5B8C_C',
            '5C0N_A', '5C7X_A', '5CZV_A', '5D93_A', '5DHV_M', '5DMI_A', '5E8D_A', '5E8E_LH',
            '5EPM_C', '5H35_C', '5IKC_M', '5JW4_A', '5JZ7_A', '5JZ7_B', '5JZ7_E', '5KN5_C',
            '5L0Q_A', '5LQB_A', '5MES_A', '5T5F_A', '5TZ2_C', '1CZ8_VW_Jack', '1CZ8_VW_FuC_5.25',
            '1JPS_T', '1KB5_AB', '1MHP_A', '1NCA_N', '1OAZ_A', '1QFW_AB', '1S78_B', '1NCB_N',
            '1N8Z_C', '2ARJ_RQ', '2FD6_AU', '2Q8A_A', '2R0K_A', '2R4R_A', '2WUC_I', '2XQB_A',
            '2YPV_A', '2ZCH_P', '3BN9_A', '3CVH_ABC', '3DVG_XY', '3L5X_A', '3L95_X', '3NCY_A',
            '3NFP_I', '3NPS_A', '3R1G_B', '3SO3_A', '3SQO_A', '3TT1_A', '3U9P_C', '3UBX_A',
            '3V6O_A', '3VG9_A', '4CAD_C', '4DKE_A', '4HC1_B', '4I18_R', '4I77_Z', '4K9E_C',
            '4LQF_A', '4LU5_B', '4M7L_T', '4MXV_B', '4OII_A', '4QEX_A', '4QWW_A', '4RGM_S',
            '4U1G_A', '4U6V_A', '4Y5V_C', '4YUE_C', '4ZSO_E', '5BVP_I', '5DFV_A', '5E94_G',
            '5EII_G', '5EU7_A', '5EZO_A', '5F3B_C', '5FB8_C', '5HDQ_A', '5HI4_B', '5J13_A',
            '5KTE_A', '5TH9_A', '5TLJ_X'
        ]

        return workloads

    def __del__(self):
        if hasattr(self, "docker_client"):
            self.docker_client.close()


if __name__ == "__main__":
    manager = AbsolutContainerManager()
    energy = manager.predict_energy("S3A3_A", "CARAAHKLARI")
    print("Predicted energy:", energy)