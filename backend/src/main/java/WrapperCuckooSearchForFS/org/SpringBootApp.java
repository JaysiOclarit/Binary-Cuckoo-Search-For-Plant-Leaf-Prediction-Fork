package WrapperCuckooSearchForFS.org;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class SpringBootApp {
    public static void main(String[] args) {
        // This launches your backend server on port 8080
        SpringApplication.run(SpringBootApp.class, args);
        System.out.println("🌱 Plant Classifier API is running on http://localhost:8080");
    }
}