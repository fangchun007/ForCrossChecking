/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // Set the number of particles
    num_particles = 100;
    particles.resize(num_particles);
    weights.resize(num_particles, 1); // set all weights to 1
    
    // Creates normal (Gaussian) distributions for x, y, and psi
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]); // std[0]: standard deviation for x
    normal_distribution<double> dist_y(y, std[1]); // std[1]: standard deviation for y
    normal_distribution<double> dist_theta(theta, std[2]); // std[2]: standard deviation for theta
    
    // Initialize particles based on estimates of x, y, theta and their uncertainty from GPS
    for (int i=0; i<num_particles; i++)
    {
        Particle p; // Create new particle
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = weights[i];
        particles[i] = p;
    }
    
    is_initialized = true;
    cout << "Particle Filter Initialization Completed" << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    // Create normal (Gaussian) distributions for x, y, and theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    // To avoid repeated conditional statement judgement, we put the same
    // judgement "yaw_rate == 0" to the outside of for-loop. As a result,
    // the length of code will be longer.
    if (abs(yaw_rate) < 0.0001)
    {
        for (int i=0; i<num_particles; i++)
        {
            // Initial location
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            double theta0 = particles[i].theta;
            // Estimate the particle's location after time delta_t
            double new_x, new_y;
            new_x = x0 + velocity * delta_t * cos(theta0);
            new_y = y0 + velocity * delta_t * sin(theta0);
            // Add random Gaussian noise
            particles[i].x = new_x + dist_x(gen);
            particles[i].y = new_y + dist_y(gen);
            particles[i].theta = particles[i].theta + dist_theta(gen);
        }
    }
    else
    {
        for (int i=0; i<num_particles; i++)
        {
            // Initial location
            double x0 = particles[i].x;
            double y0 = particles[i].y;
            double theta0 = particles[i].theta;
            // Estimate the particle's location after time delta_t
            double new_x, new_y, new_theta;
            new_x = x0 + (velocity/yaw_rate) * (sin(theta0+yaw_rate*delta_t) - sin(theta0));
            new_y = y0 + (velocity/yaw_rate) * (cos(theta0) - cos(theta0+yaw_rate*delta_t));
            new_theta = theta0 + yaw_rate*delta_t;
            // Add random Gaussian noise
            particles[i].x = new_x + dist_x(gen);
            particles[i].y = new_y + dist_y(gen);
            particles[i].theta = new_theta + dist_theta(gen);
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    int closest_id;
    double dist_min;
    double dist_pred_to_obs;
    
    for (int i=0; i<observations.size(); i++)
    {
        dist_min = 10000; // should be decided by the size of the map
        for (int j=0; j<predicted.size(); j++)
        {
            dist_pred_to_obs = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            if (dist_pred_to_obs < dist_min)
            {
                dist_min = dist_pred_to_obs;
                closest_id = j;
            }
        }
        observations[i].id = closest_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    // NOTE 1. std_landmark[] contain the Landmark measurement uncertainty in x and y direction, not the value
    // described in particle_filter.h. We can verify this point by checking it application in main.cpp.
    
    // Obtain the landmark measurement uncertainty in both x and y direction
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    
    // Set some precalculated items to avoid repeated work
    double denominator = 2*M_PI*std_x*std_y;
    double std_x2 = 2*std_x*std_x;
    double std_y2 = 2*std_y*std_y;
    
    for (int i=0; i<num_particles; i++)
    {
        Particle p = particles[i];
        double new_weight = 1.0;
        for (int j=0; j<observations.size(); j++)
        {
            LandmarkObs o = observations[j];
            // Express this observation in the MAP's coordinate system
            // The following formulas can be easily deduced by the addition and subtraction of vectors
            double x_t = p.x + o.x*cos(p.theta) - o.y*sin(p.theta);
            double y_t = p.y + o.x*sin(p.theta) + o.y*cos(p.theta);
            // Find A nearest landmark to the transformed observation
            double dist_min = sensor_range;
            double x_diff = sensor_range;
            double y_diff = sensor_range;
            for (int k=0; k<map_landmarks.landmark_list.size(); k++)
            {
                Map::single_landmark_s l = map_landmarks.landmark_list[k];
                double dist_o_to_l = dist(x_t, y_t, l.x_f, l.y_f);
                if (dist_o_to_l < dist_min)
                {
                    dist_min = dist_o_to_l;
                    x_diff = x_t - l.x_f;
                    y_diff = y_t - l.y_f;
                }
            }
            // Update weight
            new_weight = new_weight * exp(- x_diff*x_diff/std_x2 - y_diff*y_diff/std_y2) / denominator;
        }
        // Set weight
        weights[i] = new_weight;
        particles[i].weight = new_weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> resampled_p(num_particles);
    std::random_device rd;
    std::mt19937 gen(rd());
    discrete_distribution<> d(weights.begin(), weights.end());
    // Sample num_particles particles
    for (int i=0; i<num_particles; i++)
    {
        int index = d(gen);
        resampled_p[i] = particles[index];
    }
    particles = resampled_p;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
